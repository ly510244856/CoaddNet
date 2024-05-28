'''
Author: ly510244856 846149510@qq.com
Date: 2024-03-12 23:10:19
LastEditors: ly510244856 846149510@qq.com
LastEditTime: 2024-03-12 23:59:09
FilePath: /NAFNet-main/basicsr/test_speed.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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

# for test speed &latency
import time
from tqdm import tqdm

torch.autograd.set_grad_enabled(False)
T0 = 5
T1 = 10

def compute_latency_ms_pytorch(model, batch_size, resolution=224, iterations=None, device=None):
    torch.backends.cudnn.enabled = True # 启用cuDNN
    torch.backends.cudnn.benchmark = True # cuDNN 将自动寻找最优的算法来处理网络的前向和后向传播

    model.eval()
    model = model.cuda()

    input = torch.randn(batch_size, 1, resolution, resolution, device=device)

    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in tqdm(range(iterations)):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
        FPS = iterations / elapsed_time
    torch.cuda.empty_cache()
    # FPS = 1000 / latency (in ms)
    print(device, f"{latency:.4f}", 'ms @ batch size', batch_size)
    print(device, FPS, 'images/s @ batch size', batch_size)
    return latency

def throughput(model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 1, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    while time.time() - start < T0:
        model(inputs)
    timing = []
    torch.cuda.synchronize()
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        torch.cuda.synchronize()
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(device, f"{batch_size / timing.mean().item():.4f}",
          'images/s @ batch size', batch_size)

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create model
    model = create_model(opt)
    print(model.net_g)

    batch_size = 32
    resolution = 256
    torch.cuda.empty_cache()

    # 将模型和输入数据移至相同的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.net_g.to(device)

    # 计算并打印延时
    throughput(model, device, batch_size, resolution=resolution)
    compute_latency_ms_pytorch(model, batch_size=1, resolution=resolution, device=device)


if __name__ == '__main__':
    main()
