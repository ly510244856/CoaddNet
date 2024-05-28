# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import logging
import torch
from os import path as osp

import sys
sys.path.append(".")

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
from torch.nn.parallel import DataParallel, DistributedDataParallel

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if 'test' in phase:
            dataset_opt['phase'] = 'test'
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt)

    for m in model.net_g.modules():
        if hasattr(m, 'reparameterize_unireplknet'):
            m.reparameterize_unireplknet()
        
    # print(model.net_g)

    # =================================
    # save model
    # =================================
    param_key = 'params'
    model = model.net_g if isinstance(model.net_g, list) else [model.net_g]
    param_key = param_key if isinstance(param_key, list) else [param_key]
    assert len(model) == len(
        param_key), 'The lengths of net and param_key should be the same.'
    
    def get_bare_model(net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    save_dict = {}
    for net_, param_key_ in zip(model, param_key):
        net_ = get_bare_model(net_)
        state_dict = net_.state_dict()
        for key, param in state_dict.items():
            if key.startswith('module.'):  # remove unnecessary 'module.'
                key = key[7:]
            state_dict[key] = param.cpu()
        save_dict[param_key_] = state_dict

    torch.save(save_dict, 
               f"{osp.splitext(opt['path']['pretrain_network_g'])[0]}_repara.pth")
    
    opt['path']['pretrain_network_g'] = f"{osp.splitext(opt['path']['pretrain_network_g'])[0]}_repara.pth"
    opt['network_g']['deploy'] = True
    
    # create model
    model = create_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        # wheather use uint8 image to compute metrics
        use_image = opt['val'].get('use_image', True)
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'],
            rgb2bgr=rgb2bgr, use_image=use_image)

if __name__ == '__main__':
    main()
