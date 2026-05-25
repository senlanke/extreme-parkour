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

import sys  # 导入系统模块，用于在 viewer 关闭或收到退出事件时直接终止程序。
from isaacgym import gymapi  # 导入 Isaac Gym 的核心 API，用于创建仿真、viewer、相机属性等对象。
from isaacgym import gymutil, gymtorch  # 导入 Isaac Gym 工具模块和 PyTorch 张量桥接模块，用于解析设备和处理 GPU tensor。
import numpy as np  # 导入 NumPy；本文件中未直接使用，可能保留自通用任务模板。
import torch  # 导入 PyTorch，用于创建强化学习环境的 observation、reward、reset 等张量 buffer。
import time  # 导入时间模块，用于 viewer 暂停循环中短暂 sleep，避免空转占满 CPU。

# Base class for RL tasks
class BaseTask():  # 定义所有 Isaac Gym 强化学习任务的基础类，提供仿真、viewer、buffer 和通用接口骨架。

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):  # 初始化基础任务对象，接收配置、仿真参数、物理引擎、设备和是否无界面运行。
        self.gym = gymapi.acquire_gym()  # 获取 Isaac Gym 全局 gym 句柄，后续所有仿真、viewer、资产操作都通过它完成。

        self.sim_params = sim_params  # 保存仿真参数对象，例如 dt、PhysX 参数、是否使用 GPU pipeline 等。
        self.physics_engine = physics_engine  # 保存物理引擎类型，通常是 Isaac Gym 的 PhysX。
        self.sim_device = sim_device  # 保存仿真设备字符串，例如 cuda:0 或 cpu。
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)  # 解析设备字符串，得到设备类型和设备编号。
        self.headless = headless  # 保存是否无图形界面运行；训练时通常为 True，播放可视化时通常为 False。

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:  # 只有仿真在 CUDA 且启用 GPU pipeline 时，环境张量才直接放在 GPU 上。
            self.device = self.sim_device  # 将环境张量设备设置为仿真 CUDA 设备，避免频繁 CPU/GPU 拷贝。
        else:
            self.device = 'cpu'  # 否则环境张量放在 CPU 上，因为 PhysX 返回的数据会被拷贝到 CPU。

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id  # 默认让图形渲染设备和仿真设备使用同一个 GPU 编号。
        if self.headless == True:  # 如果以无界面模式运行，就不创建图形渲染设备。
            self.graphics_device_id = -1  # Isaac Gym 中 graphics_device_id=-1 表示禁用渲染输出。

        self.num_envs = cfg.env.num_envs  # 从配置中读取并行环境数量，即一次同时仿真的机器人数量。
        self.num_obs = cfg.env.num_observations  # 从配置中读取每个环境的 observation 维度。
        self.num_privileged_obs = cfg.env.num_privileged_obs  # 从配置中读取 privileged observation 维度，None 表示不单独提供 critic 特权观测。
        self.num_actions = cfg.env.num_actions  # 从配置中读取 action 维度，例如四足机器人通常是 12 个关节动作。
        
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)  # 关闭 PyTorch JIT profiling mode，减少旧版本 PyTorch/Isaac Gym 组合下的额外开销或兼容问题。
        torch._C._jit_set_profiling_executor(False)  # 关闭 PyTorch JIT profiling executor，使运行行为更稳定、少一些动态编译干扰。

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)  # 分配 observation buffer，形状为 [并行环境数, 观测维度]。
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)  # 分配 reward buffer，每个并行环境对应一个标量奖励。
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)  # 分配 reset/done buffer，初始为 1 表示环境刚创建后需要 reset。
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 分配 episode 长度计数器，记录每个环境当前 episode 已走多少步。
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # 分配超时标记 buffer，用来区分 done 是否由时间上限触发。
        if self.num_privileged_obs is not None:  # 如果配置指定了 privileged observation，就为 critic 或非对称训练额外分配 buffer。
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)  # 分配 privileged observation buffer，形状为 [并行环境数, 特权观测维度]。
        else: 
            self.privileged_obs_buf = None  # 如果没有 privileged observation，就用 None 表示算法端应直接使用普通 observation。
            # self.num_privileged_obs = self.num_obs

        self.extras = {}  # 创建额外信息字典，用于向训练算法传递 episode 统计、timeout、调试信息等非标准返回值。

        # create envs, sim and viewer
        self.create_sim()  # 调用子类实现的 create_sim() 创建 Isaac Gym 仿真、地形、机器人和并行环境。
        self.gym.prepare_sim(self.sim)  # 让 Isaac Gym 准备仿真，通常会初始化内部 tensor、物理状态和 GPU pipeline。

        # todo: read from config
        self.enable_viewer_sync = True  # 控制 viewer 是否同步仿真帧率；True 时可视化更平滑但可能变慢。
        self.viewer = None  # 初始化 viewer 句柄为空，只有非 headless 模式才会创建图形窗口。

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:  # 如果不是无界面模式，则创建 viewer 并注册键盘交互。
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(  # 创建 Isaac Gym viewer 窗口，用于交互式观察仿真。
                self.sim, gymapi.CameraProperties())  # 为 viewer 指定所属仿真和默认相机属性。
            self.gym.subscribe_viewer_keyboard_event(  # 注册 ESC 键事件，用于退出 viewer 和程序。
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")  # 将 ESC 按键映射成名为 QUIT 的 viewer action。
            self.gym.subscribe_viewer_keyboard_event(  # 注册 V 键事件，用于切换 viewer 是否同步仿真帧率。
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")  # 将 V 按键映射成 toggle_viewer_sync action。
            self.gym.subscribe_viewer_keyboard_event(  # 注册 F 键事件，用于切换自由相机和跟随相机。
                self.viewer, gymapi.KEY_F, "free_cam")  # 将 F 按键映射成 free_cam action。
            for i in range(9):  # 注册数字键 0 到 8，用于快速切换跟随查看的机器人编号。
                self.gym.subscribe_viewer_keyboard_event(  # 为当前数字键注册 viewer 键盘事件。
                self.viewer, getattr(gymapi, "KEY_"+str(i)), "lookat"+str(i))  # 将数字键映射成 lookat0 到 lookat8，用于查看对应环境中的机器人。
            self.gym.subscribe_viewer_keyboard_event(  # 注册左中括号键，用于切换到上一个机器人。
                self.viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")  # 将 [ 键映射成 prev_id action。
            self.gym.subscribe_viewer_keyboard_event(  # 注册右中括号键，用于切换到下一个机器人。
                self.viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")  # 将 ] 键映射成 next_id action。
            self.gym.subscribe_viewer_keyboard_event(  # 注册空格键，用于暂停或继续仿真显示。
                self.viewer, gymapi.KEY_SPACE, "pause")  # 将 Space 键映射成 pause action。
            self.gym.subscribe_viewer_keyboard_event(  # 注册 W 键，用于给当前查看机器人增加前向速度指令。
                self.viewer, gymapi.KEY_W, "vx_plus")  # 将 W 键映射成 vx_plus action。
            self.gym.subscribe_viewer_keyboard_event(  # 注册 S 键，用于给当前查看机器人减少前向速度指令。
                self.viewer, gymapi.KEY_S, "vx_minus")  # 将 S 键映射成 vx_minus action。
            self.gym.subscribe_viewer_keyboard_event(  # 注册 A 键，用于给当前查看机器人增加左转/航向指令。
                self.viewer, gymapi.KEY_A, "left_turn")  # 将 A 键映射成 left_turn action。
            self.gym.subscribe_viewer_keyboard_event(  # 注册 D 键，用于给当前查看机器人增加右转/航向指令。
                self.viewer, gymapi.KEY_D, "right_turn")  # 将 D 键映射成 right_turn action。
        self.free_cam = False  # 初始化为跟随相机模式，而不是自由相机模式。
        self.lookat_id = 0  # 初始化 viewer 跟随的机器人编号为第 0 个并行环境。
        self.lookat_vec = torch.tensor([-0, 2, 1], requires_grad=False, device=self.device)  # 保存相机相对机器人根位置的偏移向量，用于跟随相机视角。

    def get_observations(self):  # 返回当前 observation buffer，供算法 runner 在 reset 或 rollout 前读取。
        return self.obs_buf  # 返回普通 observation 张量，形状通常是 [num_envs, num_obs]。
    
    def get_privileged_observations(self):  # 返回当前 privileged observation buffer，供非对称 actor-critic 中的 critic 使用。
        return self.privileged_obs_buf  # 如果没有配置 privileged observation，则这里返回 None。

    def reset_idx(self, env_ids):  # 定义按环境编号重置部分环境的接口，具体机器人重置逻辑由子类实现。
        """Reset selected robots"""  # 说明该函数用于重置指定编号的机器人环境。
        raise NotImplementedError  # 基类不实现具体 reset 行为，强制子类覆盖该方法。

    def reset(self):  # 定义重置所有并行环境的通用接口。
        """ Reset all robots"""  # 说明该函数用于重置所有机器人环境。
        self.reset_idx(torch.arange(self.num_envs, device=self.device))  # 构造所有环境编号，并调用子类 reset_idx() 重置全部环境。
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))  # 用全零动作推进一步，让环境刷新观测、奖励、done 等 buffer。
        return obs, privileged_obs  # 返回 reset 后的普通观测和特权观测，供算法初始化 rollout。

    def step(self, actions):  # 定义环境推进一步的接口，输入是策略输出的 action 张量。
        raise NotImplementedError  # 基类不知道具体物理控制和奖励计算逻辑，必须由子类实现。

    def lookat(self, i):  # 将 viewer 相机切换为跟随第 i 个环境中的机器人。
        look_at_pos = self.root_states[i, :3].clone()  # 读取第 i 个机器人根状态的 xyz 位置，作为相机观察目标点。
        cam_pos = look_at_pos + self.lookat_vec  # 根据保存的相机偏移向量计算相机当前位置。
        self.set_camera(cam_pos, look_at_pos)  # 调用子类或环境实现的 set_camera() 设置 viewer 相机位置和朝向目标。

    def render(self, sync_frame_time=True):  # 渲染 viewer 并处理键盘事件，sync_frame_time 控制是否按仿真帧率同步显示。
        if self.viewer:  # 只有创建了 viewer 时才执行渲染和交互逻辑，headless 训练时这里通常跳过。
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):  # 检查 viewer 窗口是否已经被用户关闭。
                sys.exit()  # 如果 viewer 已关闭，直接退出程序，避免继续访问无效窗口。
            if not self.free_cam:  # 如果当前不是自由相机模式，就持续跟随指定机器人。
                self.lookat(self.lookat_id)  # 更新相机位置，使其看向当前 lookat_id 对应的机器人。
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):  # 遍历本帧 viewer 收到的所有键盘 action 事件。
                if evt.action == "QUIT" and evt.value > 0:  # 如果收到退出事件且按键处于触发状态。
                    sys.exit()  # 立即退出程序。
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:  # 如果收到 viewer 同步开关事件且按键触发。
                    self.enable_viewer_sync = not self.enable_viewer_sync  # 在同步和非同步渲染模式之间切换。
                
                if not self.free_cam:  # 只有跟随相机模式下，数字切换和指令键才作用于当前跟随机器人。
                    for i in range(9):  # 遍历数字键 0 到 8 对应的可快速查看机器人编号。
                        if evt.action == "lookat" + str(i) and evt.value > 0:  # 如果按下了对应数字键。
                            self.lookat(i)  # 立即把相机切换到第 i 个机器人。
                            self.lookat_id = i  # 更新当前跟随机器人编号。
                    if evt.action == "prev_id" and evt.value > 0:  # 如果按下了切换到上一个机器人的按键。
                        self.lookat_id  = (self.lookat_id-1) % self.num_envs  # 环形递减当前跟随编号，避免越界。
                        self.lookat(self.lookat_id)  # 把相机切换到新的跟随机器人。
                    if evt.action == "next_id" and evt.value > 0:  # 如果按下了切换到下一个机器人的按键。
                        self.lookat_id  = (self.lookat_id+1) % self.num_envs  # 环形递增当前跟随编号，避免越界。
                        self.lookat(self.lookat_id)  # 把相机切换到新的跟随机器人。
                    if evt.action == "vx_plus" and evt.value > 0:  # 如果按下 W 键增加前向速度指令。
                        self.commands[self.lookat_id, 0] += 0.2  # 给当前查看机器人对应的 x 方向速度命令增加 0.2。
                    if evt.action == "vx_minus" and evt.value > 0:  # 如果按下 S 键减少前向速度指令。
                        self.commands[self.lookat_id, 0] -= 0.2  # 给当前查看机器人对应的 x 方向速度命令减少 0.2。
                    if evt.action == "left_turn" and evt.value > 0:  # 如果按下 A 键增加左转或目标 heading 指令。
                        self.commands[self.lookat_id, 3] += 0.5  # 修改当前查看机器人的第 4 个 command 分量，通常对应 heading。
                    if evt.action == "right_turn" and evt.value > 0:  # 如果按下 D 键增加右转或反向 heading 指令。
                        self.commands[self.lookat_id, 3] -= 0.5  # 修改当前查看机器人的第 4 个 command 分量，通常对应 heading。
                if evt.action == "free_cam" and evt.value > 0:  # 如果按下 F 键切换相机模式。
                    self.free_cam = not self.free_cam  # 在自由相机和跟随相机之间切换。
                    if self.free_cam:  # 如果刚切换到自由相机模式。
                        self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)  # 将相机设置到配置中定义的默认自由视角。
                
                if evt.action == "pause" and evt.value > 0:  # 如果按下空格键触发暂停。
                    self.pause = True  # 设置暂停标记，进入暂停循环。
                    while self.pause:  # 在暂停状态下停留，直到再次收到 pause 事件。
                        time.sleep(0.1)  # 每次循环睡眠 0.1 秒，避免暂停时 CPU 忙等。
                        self.gym.draw_viewer(self.viewer, self.sim, True)  # 在暂停期间继续绘制 viewer，让窗口保持响应和显示。
                        for evt in self.gym.query_viewer_action_events(self.viewer):  # 检查暂停期间是否有新的键盘事件。
                            if evt.action == "pause" and evt.value > 0:  # 如果再次按下空格键。
                                self.pause = False  # 退出暂停循环，恢复仿真。
                        if self.gym.query_viewer_has_closed(self.viewer):  # 暂停期间也检查 viewer 是否被关闭。
                            sys.exit()  # 如果窗口被关闭，直接退出程序。

                        
                
            # fetch results
            if self.device != 'cpu':  # 如果环境张量在 GPU 上，渲染前需要确保仿真结果已经完成并可读取。
                self.gym.fetch_results(self.sim, True)  # 阻塞等待当前仿真步完成，并获取物理结果。

            self.gym.poll_viewer_events(self.viewer)  # 轮询 viewer 窗口系统事件，保持窗口响应。
            # step graphics
            if self.enable_viewer_sync:  # 如果启用 viewer 同步，就按正常图形流程推进并绘制一帧。
                self.gym.step_graphics(self.sim)  # 推进图形渲染管线，更新可视化状态。
                self.gym.draw_viewer(self.viewer, self.sim, True)  # 将当前仿真画面绘制到 viewer 窗口。
                if sync_frame_time:  # 如果要求同步帧时间。
                    self.gym.sync_frame_time(self.sim)  # 按仿真时间步同步显示速度，避免 viewer 播放过快。
            else:
                self.gym.poll_viewer_events(self.viewer)  # 非同步模式下只轮询窗口事件，不强制图形帧同步。
            
            if not self.free_cam:  # 如果当前是跟随相机模式，就更新相机相对机器人位置的偏移量。
                p = self.gym.get_viewer_camera_transform(self.viewer, None).p  # 读取 viewer 当前相机的世界坐标位置。
                cam_trans = torch.tensor([p.x, p.y, p.z], requires_grad=False, device=self.device)  # 将相机位置转换成和环境同设备的 PyTorch 张量。
                look_at_pos = self.root_states[self.lookat_id, :3].clone()  # 读取当前跟随机器人的根位置。
                self.lookat_vec = cam_trans - look_at_pos  # 保存相机相对机器人根位置的偏移，下一帧继续用于跟随视角。
            
