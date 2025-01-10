import json
from os.path import join, abspath, dirname

from settings.defaults import _C
from settings.setup_functions import *

"""
训练代码：
setup.py:
    1、修改cfg_file指向的yaml文件
    2、修改config.misc.log_name
    3、修改config.cuda_visible
    4、修改config.model.type
    5、是否使用baseline模型
    6、是否使用timm库中的模型
    7、修改模型描述
build.py:
    7、修改build.py中指向的模型
config.yaml:
    8、修改yaml文件中的batch_size、lr等参数
"""

root = dirname(abspath(__file__))
config = _C.clone()
cfg_file = join(root, 'configs', 'swin-cub.yaml')
config = SetupConfig(config, cfg_file)
config.defrost()

# Log Name and Perferences
config.write = True
config.train.checkpoint = True
config.misc.exp_name = f'{config.data.dataset}'

config.misc.log_name = f'Ours'
config.cuda_visible = '0,1'
config.model.type = "mymodel"
# 是否使用baseline模型
config.model.baseline_model = False
# 是否使用timm库中的模型
config.model.timm = True
"""模型描述"""
config.model.description = "相比于exp1,修改mask_cross_attn的方式为softmax_with_mask,"

# Environment Settings，同时用作可视化路径
config.data.log_path = join(config.misc.output, config.misc.exp_name, config.misc.log_name
                            + time.strftime(' %m-%d_%H-%M', time.localtime()))

config.model.pretrained = join(
    config.model.pretrained,
    config.model.name + config.model.pre_version + config.model.pre_suffix)
os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible
os.environ['OMP_NUM_THREADS'] = '1'

# Setup Functions
config.nprocess, config.local_rank = SetupDevice()
config.data.data_root, config.data.batch_size = LocateDatasets(config)
config.train.lr = ScaleLr(config)
log = SetupLogs(config, config.local_rank)
if config.write and config.local_rank in [-1, 0]:
    with open(config.data.log_path + '/config.json', "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        # f.write(config.dump())
config.freeze()
SetSeed(config)
