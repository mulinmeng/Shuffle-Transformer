import numpy as np
import json
from shutil import copyfile
import os


rd_data = "%05d" % np.random.randint(99999)
USER_DIR = "/apdcephfs/share_1227775/wqiangzhang/"

def create_task_flag(name, host_num=1, gpu_num=1, batch_size=1):
    task_flag = "h%d_bs%dx%d_%s_%s" % (host_num, gpu_num, batch_size, name, rd_data)
    return task_flag

################### modify #############################################################
CLUSTER_CFG = {
    "host_num": 1,
    "host_gpu_num": 8
}

readable_name = 'shuffle-transformer'

json_cfg=dict(
    REQUIRED=dict(
        task_flag = create_task_flag(
            readable_name, 
            CLUSTER_CFG['host_num'], CLUSTER_CFG['host_gpu_num'], 
            batch_size=80),
        readable_name = readable_name,
        host_num = CLUSTER_CFG['host_num'],
        host_gpu_num=CLUSTER_CFG['host_gpu_num'],
        model_local_file_path = os.path.join(USER_DIR, 'Shuffle-Transformer/code'),
    ),
    OPTIONAL=dict(
        is_elasticity=True
    )
)

################### modify ###############################################################

def creat_submit_file(json_cfg):
    config_name = 'job_cfg' + str(np.random.randint(99999)) +'.json'
    copyfile('./config.json', config_name)

    with open(config_name, 'r') as f:
        cfg = json.load(f)
        # REQUIRED
        cfg.update(json_cfg['REQUIRED'])
        cfg.update(json_cfg['OPTIONAL'])
    
    with open(config_name, 'w') as f:
        json.dump(cfg, f)
        
    return config_name


def submit():
    config_name = creat_submit_file(json_cfg)
    os.system('jizhi_client start -scfg ' + config_name)
    os.system('rm ' + config_name)


if __name__ == "__main__":
    submit()
