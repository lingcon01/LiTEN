import os
import sys
import random
import logging
import argparse
import yaml
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import optim
from torch_scatter import scatter
from torch_geometric.data import DataLoader
import numpy as np
from easydict import EasyDict
from tqdm import tqdm

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.LiTEN import LiTEN
from dataset.chignolin import Chignolin, get_mean_std

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='MD17 Training Script')
    parser.add_argument('--config_path', type=str, default=root_dir + '/config/chignolin.yml',
                        help='Path to config yaml file.')
    parser.add_argument('--molecule', type=str, default='chignolin', help='Molecule dataset to use.')
    parser.add_argument('--restore_path', type=str, default='', help='Checkpoint restore path.')
    parser.add_argument('--save_path', type=str, default=root_dir + '/ckpt',
                        help='Checkpoint save directory.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu).')
    return parser.parse_args()


def load_config(config_path, args):
    """Load config file and override with command-line args."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config = EasyDict(config)

    if args.molecule:
        config.data.molecule = args.molecule
    if args.restore_path:
        config.train.restore_path = args.restore_path
    if args.save_path:
        config.train.save_path = args.save_path
    if config.data.base_path is None:
        config.data.base_path = root_dir + '/data'

    # Construct dataset path
    config.data.base_path = os.path.join(config.data.base_path, config.data.molecule)

    # Create save directory if not exists
    os.makedirs(os.path.join(config.train.save_path, config.data.molecule), exist_ok=True)

    return config


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model, ckpt_path, device, single=True):
    """Load model weights, handling multi-GPU 'module.' prefix if present."""
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get('model', state)

    if single:
        # Remove 'module.' prefix from multi-GPU training if present
        new_state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
    else:
        new_state_dict = state_dict

    model_state_dict = model.state_dict()
    loaded_params = 0
    for k, v in model_state_dict.items():
        if k in new_state_dict and new_state_dict[k].shape == v.shape:
            model_state_dict[k] = new_state_dict[k]
            loaded_params += 1

    if loaded_params > 20:
        logger.info('Checkpoint loaded successfully.')

    model.load_state_dict(model_state_dict, strict=True)
    return model


def train_one_epoch(epoch, dataloader, model, optimizer, config, device, mean, std, partition='train'):
    """
    Run one epoch of training or validation.
    Args:
        partition: 'train' or 'valid'
    """
    if partition == 'train':
        model.train()
    else:
        model.eval()

    record = {
        'loss_sum': 0,
        'count': 0,
        'loss_list': [],
        'energy_sum': 0,
        'energy_list': [],
        'force_sum': 0,
        'force_list': []
    }

    # Weight scheduling
    if epoch < (config.train.epochs * 3 // 4):
        energy_weight = config.train.energy_weight
        force_weight = config.train.force_weight
        momentum_weight = config.train.momentum_weight
    else:
        energy_weight = config.train.swa_energy_weight
        force_weight = config.train.swa_force_weight
        momentum_weight = config.train.momentum_weight
        logger.info(f'Starting SWA: force_weight={force_weight}, energy_weight={energy_weight}')

    is_train = (partition == 'train')

    for i, data in tqdm(enumerate(dataloader), unit='batch', mininterval=120):
        if is_train:
            optimizer.zero_grad()

        # Prepare data
        label = data.y.squeeze(-1).to(device)
        assert label.dim() == 1, f"Expected 1D label but got {label.dim()}D with shape {label.shape}"

        dy = data.dy.to(device)
        data.batch = data.batch.to(device)
        data.pos = data.pos.to(device)
        data.z = data.z.to(device)

        total_force = scatter(dy, data.batch, dim=0, reduce='mean')

        pred, pdy = model(data, mean, std, is_train)

        assert label.dim() == pred.dim(), "Prediction and label dimension mismatch"

        total_force_pred = scatter(pdy, data.batch, dim=0, reduce='mean')

        batch_size = len(pred)

        # Calculate losses
        loss_energy = F.l1_loss(pred, label)
        loss_force = F.l1_loss(dy, pdy)
        loss_momentum = F.l1_loss(total_force_pred, total_force)  # Currently unused

        if is_train:
            loss = loss_energy * energy_weight + loss_force * force_weight
            loss.backward()
            if not torch.isnan(loss):
                optimizer.step()
        else:
            loss = loss_energy * energy_weight + loss_force * force_weight

        # Record statistics
        record['loss_sum'] += loss.item() * batch_size
        record['energy_sum'] += loss_energy.item() * batch_size
        record['force_sum'] += loss_force.item() * batch_size
        record['count'] += batch_size
        record['loss_list'].append(loss.item())
        record['energy_list'].append(loss_energy.item())
        record['force_list'].append(loss_force.item())

        # Log progress
        if i % config.train.log_interval == 0:
            logger.info(
                f'Epoch {epoch:4d} | Iter {i:4d} | '
                f'Loss {np.mean(record["loss_list"][-10:]):.4f} | '
                f'Energy {np.mean(record["energy_list"][-10:]):.4f} | '
                f'Force {np.mean(record["force_list"][-10:]):.4f} | '
                f'LR {optimizer.param_groups[0]["lr"]:.7f}'
            )

    return record['loss_sum'] / record['count'], record['energy_sum'] / record['count'], record['force_sum'] / record[
        'count']


def save_checkpoint(state, checkpoint_list, max_ckpts, save_path):
    """
    Save checkpoint and keep only a limited number of saved files.
    """
    if len(checkpoint_list) >= max_ckpts:
        try:
            os.remove(checkpoint_list[0])
            logger.info(f'Removed old checkpoint {checkpoint_list[0]}')
        except Exception as e:
            logger.warning(f'Failed to remove checkpoint {checkpoint_list[0]}: {e}')
        checkpoint_list.pop(0)

    checkpoint_list.append(save_path)
    torch.save(state, save_path)


def main():
    args = parse_args()
    config = load_config(args.config_path, args)

    logger.info(f'Training molecule: {config.data.molecule} | '
                f'Force weight: {config.train.force_weight} | '
                f'Energy weight: {config.train.energy_weight}')

    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    logger.info(f'Using device: {device}')

    set_seed(config.train.seed)

    # 加载数据集
    dataset = Chignolin(config.data.base_path)
    size = len(dataset)
    print(f'The dataset contain {size} molecule')
    idx = np.arange(size)
    np.random.shuffle(idx)

    train_idx = idx[:config.data.num_train]
    val_idx = idx[config.data.num_train: config.data.num_train + config.data.num_val]
    train_set = dataset[train_idx]
    val_set = dataset[val_idx]

    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True,
                              num_workers=config.train.num_workers, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=config.test.test_batch_size, shuffle=False,
                            num_workers=config.train.num_workers, pin_memory=True)

    mean, std = get_mean_std(train_loader)

    # Initialize model
    model = LiTEN(num_heads=config.model.num_heads,
                  num_layers=config.model.num_layers,
                  hidden_channels=config.model.hidden_channels,
                  num_rbf=config.model.num_rbf,
                  cutoff=config.model.cutoff,
                  max_neighbors=config.model.max_neighbors,
                  max_z=config.model.max_z,
                  vec_norm=config.model.vec_norm).to(device)

    logger.info(model)
    logger.info(f'Model parameters count: {sum(p.numel() for p in model.parameters())}')

    # Load pretrained model if provided
    if config.train.restore_path:
        model = load_model(model, config.train.restore_path, device)
        logger.info(f'Model restored from {config.train.restore_path}')

    # Optimizer and LR scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=float(config.train.weight_decay))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.train.factor,
        patience=config.train.patience,
        min_lr=float(config.train.min_lr),
    )

    # Training loop
    best_results = {'energy': 1e10, 'force': 1e10, 'epoch': 0}
    energy_ckpts, force_ckpts = [], []

    for epoch in range(config.train.epochs):
        train_loss, train_energy, train_force = train_one_epoch(
            epoch, train_loader, model, optimizer, config, device, mean, std, partition='train'
        )

        if epoch % config.test.test_interval == 0:
            val_loss, val_energy, val_force = train_one_epoch(
                epoch, val_loader, model, optimizer, config, device, mean, std, partition='valid'
            )

            lr_scheduler.step(val_loss)

            # Save best energy checkpoint
            if val_energy < best_results['energy']:
                best_results['energy'] = val_energy
                best_results['epoch'] = epoch
                ckpt_path = os.path.join(config.train.save_path, config.data.molecule, f'energy_checkpoint_{epoch}')
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }
                save_checkpoint(state, energy_ckpts, max_ckpts=5, save_path=ckpt_path)
                logger.info(f'Best energy checkpoint saved at epoch {epoch}, val_energy={val_energy:.4f}')

            # Save best force checkpoint
            if val_force < best_results['force']:
                best_results['force'] = val_force
                best_results['epoch'] = epoch
                ckpt_path = os.path.join(config.train.save_path, config.data.molecule, f'force_checkpoint_{epoch}')
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }
                save_checkpoint(state, force_ckpts, max_ckpts=5, save_path=ckpt_path)
                logger.info(f'Best force checkpoint saved at epoch {epoch}, val_force={val_force:.4f}')

            logger.info(f'Validation - Energy Loss: {val_energy:.4f}, Force Loss: {val_force:.4f}')
            logger.info(
                f'Best Energy: {best_results["energy"]:.4f}, Best Force: {best_results["force"]:.4f}, Best Epoch: {best_results["epoch"]}')


if __name__ == '__main__':
    main()
