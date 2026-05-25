# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np  # 导入 NumPy，通常用于数值计算；此脚本中未直接使用，但保留可能是为了兼容原始训练脚本结构。
import os  # 导入操作系统接口模块，用于设置环境变量和创建日志目录。
from datetime import datetime  # 导入日期时间工具；此脚本中未直接使用，可能保留自通用训练模板。

# Avoid NVRTC architecture errors from TorchScript fusers on newer GPUs with
# the legacy PyTorch/CUDA version required by Isaac Gym.
os.environ.setdefault("PYTORCH_JIT", "0")  # 如果外部没有显式设置 PYTORCH_JIT，则默认关闭 PyTorch JIT，以避免旧 CUDA/PyTorch 与新显卡组合下的 NVRTC 编译错误。

import isaacgym  # 导入 Isaac Gym；需要在 torch 相关模块前导入，以确保 Isaac Gym 的底层库和 GPU 管线正确初始化。
from legged_gym.envs import *  # 导入并触发环境模块初始化，使 envs/__init__.py 中的任务注册逻辑生效，例如注册 a1、go1 等任务。
from legged_gym.utils import get_args, task_registry  # 导入命令行参数解析函数和全局任务注册表，用于创建环境与算法 runner。
from shutil import copyfile  # 导入文件复制工具；此脚本中未直接使用，可能保留自原始模板或用于后续扩展。
import torch  # 导入 PyTorch，训练策略网络和张量计算依赖它；此脚本中主要由下游环境和算法模块使用。
import wandb  # 导入 Weights & Biases，用于在线记录训练指标、实验配置和关键源码文件。

def train(args):
    args.headless = True  # 强制训练过程以无图形界面模式运行，避免打开 Isaac Gym viewer，从而节省显存和渲染开销。
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid  # 根据项目名和实验 ID 拼出本次训练的日志与模型保存目录。
    try:
        os.makedirs(log_pth)  # 尝试创建日志目录；如果目录不存在，会递归创建完整路径。
    except:
        pass  # 如果目录已经存在或创建失败，这里直接忽略异常，继续后续训练流程。
    if args.debug:
        mode = "disabled"  # 调试模式下关闭 wandb 在线记录，避免产生正式实验日志。
        args.rows = 10  # 调试模式下将地形行数设小，减少仿真场景规模。
        args.cols = 8  # 调试模式下将地形列数设小，加快环境构建和调试速度。
        args.num_envs = 64  # 调试模式下只创建 64 个并行环境，降低 GPU/CPU 负载。
    else:
        mode = "online"  # 非调试模式下使用 wandb 在线模式，正常上传训练指标和实验信息。
    
    if args.tensorboard:
        mode = "disabled"  # 如果用户选择 TensorBoard 记录，则关闭 wandb，避免同时使用两套在线日志系统。
    if not args.tensorboard:
        wandb.init(project=args.proj_name, name=args.exptid, entity="parkour", group=args.exptid[:3], mode=mode, dir="../../logs")  # 初始化 wandb 实验，项目名来自 --proj_name，运行名来自 --exptid，并按实验 ID 前三位分组。
        wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")  # 将基础环境配置文件上传到 wandb，方便之后复现实验时查看当时使用的配置。
        wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")  # 将核心机器人环境实现上传到 wandb，记录本次训练对应的环境逻辑。
    print(f"TensorBoard log dir: {log_pth}")  # 打印日志目录；即使使用 wandb，这个目录也用于本地保存 checkpoint 和 TensorBoard 事件文件。

    env, env_cfg = task_registry.make_env(name=args.task, args=args)  # 根据任务名创建 Isaac Gym 向量化环境，并返回实际使用的环境配置。
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, init_wandb=not args.tensorboard)  # 创建 on-policy PPO 训练器，内部会构建 actor-critic、estimator、可选深度编码器和 PPO 算法对象。
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)  # 启动训练循环，训练迭代次数来自配置，并将初始 episode 长度随机化以提升采样多样性。

if __name__ == '__main__':
    # Log configs immediately
    args = get_args()  # 解析命令行参数，并补全 Isaac Gym、仿真设备、强化学习设备等运行参数。
    train(args)  # 将解析后的参数传入训练主函数，正式开始创建环境、创建算法并执行训练。
