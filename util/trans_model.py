import sys
import torch
from collections import OrderedDict

sys.path.append('/home/suqun/model/LiTEN')
from model.LiTEN_FF import LiTEN

old_prefix = 'LumiForce'
new_prefix = 'LiTEN'

def replace_state_dict_prefix(state_dict, old_prefix, new_prefix):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(old_prefix):
            new_key = new_prefix + k[len(old_prefix):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

def main():
    # 1. load old model (requires old model class in PYTHONPATH)
    old_model_path = '/home/suqun/model/LiTEN/checkpoints/nablaDFT.model'
    old_model = torch.load(old_model_path, map_location='cpu')

    # 2. replace keys in state_dict
    old_state_dict = old_model.state_dict()
    new_state_dict = replace_state_dict_prefix(old_state_dict, old_prefix, new_prefix)

    new_model = LiTEN()  # 需要正确初始化参数

    # 4. load new state dict
    new_model.load_state_dict(new_state_dict)

    # 5. save entire new model object
    torch.save(new_model, '/home/suqun/model/LiTEN/checkpoints/LiTEN_nablaDFT.model')
    print("Saved new model to SPICE_12_LiTEN_FF.model")

if __name__ == '__main__':
    main()
