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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs  # 导入工程根目录和环境注册表，供资产路径和环境发现使用
from time import time  # 导入计时函数，用于统计地形创建耗时
from warnings import WarningMessage  # 保留 warning 类型导入，兼容上游接口
import numpy as np  # 导入 NumPy，用于随机化参数和网格计算
import os  # 导入路径工具，用于拆分机器人资产路径

from isaacgym.torch_utils import *  # 导入 Isaac Gym 张量数学工具，用于四元数和随机采样
from isaacgym.torch_utils import torch_rand_float  # 显式导入随机张量采样函数，避免静态检查器漏判星号导入
from isaacgym import gymtorch, gymapi, gymutil  # 导入 Gym API、tensor 包装和调试绘制工具

import torch, torchvision  # 导入 PyTorch 与 torchvision，支撑 GPU 训练和深度图缩放
from torch import Tensor  # 导入 Tensor 类型，保留接口标注能力
from typing import Tuple, Dict  # 导入类型标注工具，兼容配置和返回值说明

from legged_gym import LEGGED_GYM_ROOT_DIR  # 再次导入工程根目录，沿用原文件的资产路径依赖
from legged_gym.envs.base.base_task import BaseTask  # 导入基类，复用仿真生命周期和 viewer 管理
from legged_gym.utils.terrain import Terrain  # 导入地形生成器，创建 parkour 训练场景
from legged_gym.utils.math import *  # 导入项目数学工具，处理坐标变换和随机张量
from legged_gym.utils.helpers import class_to_dict  # 导入配置转换工具，便于遍历奖励和命令范围
from scipy.spatial.transform import Rotation as R  # 保留 SciPy 旋转工具，方便姿态处理扩展
from .legged_robot_config import LeggedRobotCfg  # 导入环境配置类型，明确构造函数输入结构

from tqdm import tqdm  # 导入进度条，显示并行环境创建进度
import cv2  # 导入 OpenCV，用于显示深度图调试窗口
import matplotlib.pyplot as plt  # 保留绘图库，方便后续调试可视化

def euler_from_quaternion(quat_angle):  # 定义四元数转欧拉角逻辑
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]  # 拆出四元数四个分量，后续公式按 x/y/z/w 使用
        t0 = +2.0 * (w * x + y * z)  # 计算 roll 的 atan2 分子项
        t1 = +1.0 - 2.0 * (x * x + y * y)  # 计算 roll 的 atan2 分母项
        roll_x = torch.atan2(t0, t1)  # 得到绕机身 x 轴的 roll 角

        t2 = +2.0 * (w * y - z * x)  # 计算 pitch 的 asin 输入项
        t2 = torch.clip(t2, -1, 1)  # 限制 asin 输入范围，避免数值误差越界
        pitch_y = torch.asin(t2)  # 得到绕机身 y 轴的 pitch 角

        t3 = +2.0 * (w * z + x * y)  # 计算 yaw 的 atan2 分子项
        t4 = +1.0 - 2.0 * (y * y + z * z)  # 计算 yaw 的 atan2 分母项
        yaw_z = torch.atan2(t3, t4)  # 得到绕 z 轴的 yaw 角

        return roll_x, pitch_y, yaw_z  # 返回四元数转欧拉角结果

class LeggedRobot(BaseTask):  # 定义腿式机器人环境主体，集中管理仿真、观测、奖励和 reset
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):  # 定义环境初始化逻辑
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg  # 保存环境配置，所有控制、地形、奖励参数都从这里读取
        self.sim_params = sim_params  # 保存仿真参数，创建 PhysX sim 时使用
        self.height_samples = None  # 初始化高度图缓存，地形创建后再填入真实数据
        self.debug_viz = True  # 默认打开调试绘制，方便检查目标点和足端状态
        self.init_done = False  # 标记初始化是否完成，保护课程逻辑不在首次 reset 误触发
        self._parse_cfg(self.cfg)  # 先展开配置，后续创建仿真和缓存都依赖这些标量
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)  # 调用基类初始化，完成通用任务框架设置

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]),  # 按深度编码器输入尺寸创建统一缩放算子
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)  # 使用 bicubic 缩放，减少深度图重采样锯齿

        if not self.headless:  # 仅在可视化运行时设置 viewer 相机
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)  # 设置 viewer 初始视角，方便人工观察训练场景
        self._init_buffers()  # 初始化训练循环反复复用的 Gym tensor 视图和缓存
        self._prepare_reward_function()  # 根据奖励配置注册实际会调用的奖励函数
        self.init_done = True  # 标记初始化是否完成，保护课程逻辑不在首次 reset 误触发
        self.global_counter = 0  # 累计策略步数，驱动动作延迟和相机刷新
        self.total_env_steps_counter = 0  # 累计环境交互步数，供训练统计使用

        self.reset_idx(torch.arange(self.num_envs, device=self.device))  # 首次重置所有并行环境，让状态缓存进入有效 episode
        self.post_physics_step()  # 推进物理后统一刷新状态、奖励、终止和观测

    def step(self, actions):  # 定义策略步推进逻辑
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self.reindex(actions)  # 保存策略动作在当前处理阶段的版本

        actions.to(self.device)  # 确保动作位于训练设备，避免后续 tensor 拼接跨设备
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)  # 维护动作滑动窗口，用于模拟执行延迟
        if self.cfg.domain_rand.action_delay:  # 启用动作延迟随机化时才从历史动作队列取值
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:  # 到达延迟课程更新时间点时才切换延迟步数
                if len(self.cfg.domain_rand.action_curr_step) != 0:  # 延迟课程列表还有候选值时继续推进课程
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)  # 记录当前使用的动作延迟步数
            if self.viewer:  # viewer 回放时使用固定延迟，方便可视化对比
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)  # 记录当前使用的动作延迟步数
            indices = -self.delay -1  # 把延迟步数换算成动作历史索引
            actions = self.action_history_buf[:, indices.long()]  # 保存策略动作在当前处理阶段的版本

        self.global_counter += 1  # 累计策略步数，驱动动作延迟和相机刷新
        self.total_env_steps_counter += 1  # 累计环境交互步数，供训练统计使用
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale  # 把动作裁剪阈值换算到未缩放动作空间
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)  # 保存裁剪后的动作，作为控制器输入
        self.render()  # 处理 viewer 同步和键盘事件，headless 训练时基本为空操作

        for _ in range(self.cfg.control.decimation):  # 一个策略步内重复多个物理子步
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)  # 把动作转换成当前物理子步实际执行的力矩
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))  # 把关节力矩写入仿真执行器
            self.gym.simulate(self.sim)  # 推进 PhysX 物理仿真一步
            self.gym.fetch_results(self.sim, True)  # 等待仿真完成，确保后续读取最新状态
            self.gym.refresh_dof_state_tensor(self.sim)  # 刷新关节位置和速度缓存
        self.post_physics_step()  # 推进物理后统一刷新状态、奖励、终止和观测

        clip_obs = self.cfg.normalization.clip_observations  # 读取观测裁剪范围，抑制异常值进入网络
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)  # 保存 actor 网络最终观测
        if self.privileged_obs_buf is not None:  # 只有 critic 特权观测存在时才裁剪它
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)  # 保存 critic 特权观测并保持裁剪一致
        self.extras["delta_yaw_ok"] = self.delta_yaw < 0.6  # 记录朝向误差是否处在视觉策略可用范围内
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:  # 只在启用深度相机且到达刷新周期时执行
            self.extras["depth"] = self.depth_buffer[:, -2]  # 返回上一帧稳定深度图，避开刚写入帧的同步风险
        else:  # 本步没有刷新相机时不向 runner 发送新深度图
            self.extras["depth"] = None  # 本步没有新深度图时显式告诉 runner 不更新视觉输入
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras  # 返回策略步推进结果

    def get_history_observations(self):  # 定义历史观测读取逻辑
        return self.obs_history_buf  # 返回历史观测读取结果

    def normalize_depth_image(self, depth_image):  # 定义深度图归一化逻辑
        depth_image = depth_image * -1  # 保存当前环境处理后的深度帧
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5  # 保存当前环境处理后的深度帧
        return depth_image  # 返回深度图归一化结果

    def process_depth_image(self, depth_image, env_id):  # 定义深度图预处理逻辑

        depth_image = self.crop_depth_image(depth_image)  # 保存当前环境处理后的深度帧
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]  # 保存当前环境处理后的深度帧
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)  # 保存当前环境处理后的深度帧
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()  # 保存当前环境处理后的深度帧
        depth_image = self.normalize_depth_image(depth_image)  # 保存当前环境处理后的深度帧
        return depth_image  # 返回深度图预处理结果

    def crop_depth_image(self, depth_image):  # 定义深度图裁剪逻辑

        return depth_image[:-2, 4:-4]  # 返回深度图裁剪结果

    def update_depth_buffer(self):  # 定义深度缓存刷新逻辑
        if not self.cfg.depth.use_camera:  # 只在启用深度相机且到达刷新周期时执行
            return  # 没有待处理环境时提前退出

        if self.global_counter % self.cfg.depth.update_interval != 0:  # 只在启用深度相机且到达刷新周期时执行
            return  # 没有待处理环境时提前退出
        self.gym.step_graphics(self.sim)  # 推进图形管线，headless 相机渲染也需要这一步
        self.gym.render_all_camera_sensors(self.sim)  # 渲染所有环境的相机，生成最新深度图
        self.gym.start_access_image_tensors(self.sim)  # 进入 GPU 图像 tensor 访问区间，避免和渲染写入冲突

        for i in range(self.num_envs):  # 逐个处理并行环境
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,  # 保存 Isaac Gym 返回的原始 GPU 深度 tensor
                                                                self.envs[i],  # 指定第 i 个并行环境的 Gym 句柄
                                                                self.cam_handles[i],  # 指定第 i 个环境挂载的深度相机句柄
                                                                gymapi.IMAGE_DEPTH)  # 请求深度图通道，而不是 RGB 或分割图

            depth_image = gymtorch.wrap_tensor(depth_image_)  # 保存当前环境处理后的深度帧
            depth_image = self.process_depth_image(depth_image, i)  # 保存当前环境处理后的深度帧

            init_flag = self.episode_length_buf <= 1  # 判断 episode 是否刚开始，用于初始化深度历史
            if init_flag[i]:  # episode 首帧没有历史帧，需要用当前深度图填满缓存
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)  # 分配深度图历史缓存
            else:  # 当前条件未命中时走默认处理路径
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)  # 分配深度图历史缓存

        self.gym.end_access_image_tensors(self.sim)  # 结束图像 tensor 访问区间，允许渲染管线继续更新

    def _update_goals(self):  # 定义目标推进逻辑
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt  # 标记已经稳定到达 waypoint 的环境
        self.cur_goal_idx[next_flag] += 1  # 保存每个环境当前 waypoint 索引
        self.reach_goal_timer[next_flag] = 0  # 累计停留在目标半径内的步数以做防抖

        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold  # 标记机器人是否已经进入当前目标半径
        self.reach_goal_timer[self.reached_goal_ids] += 1  # 累计停留在目标半径内的步数以做防抖

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]  # 保存当前目标相对机器人位置
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]  # 保存下一目标相对机器人位置

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)  # 计算目标向量长度，并为归一化提供分母
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)  # 保存指向目标的单位向量
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])  # 保存当前目标方向 yaw

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)  # 计算目标向量长度，并为归一化提供分母
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)  # 保存指向目标的单位向量
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])  # 保存下一目标方向 yaw

    def post_physics_step(self):  # 定义物理步后处理逻辑
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)  # 刷新 base 根状态，用于位置、姿态和速度观测
        self.gym.refresh_net_contact_force_tensor(self.sim)  # 刷新接触力，用于碰撞惩罚和足端接触判断
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # 刷新刚体状态，用于足端位置和调试绘制
        self.gym.refresh_force_sensor_tensor(self.sim)  # 刷新足端力传感器读数

        self.episode_length_buf += 1  # 记录每个环境当前 episode 长度
        self.common_step_counter += 1  # 记录全局公共步数，驱动命令和外推调度

        self.base_quat[:] = self.root_states[:, 3:7]  # 缓存 base 姿态四元数
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # 缓存机身坐标系线速度
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 缓存机身坐标系角速度
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # 缓存重力在机身坐标系下的投影
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt  # 缓存由根速度差分得到的机身加速度

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)  # 缓存欧拉角，用于终止和朝向奖励

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.  # 按足端接触力阈值得到当前帧接触状态
        self.contact_filt = torch.logical_or(contact, self.last_contacts)  # 合并当前和上一帧接触，过滤单帧抖动
        self.last_contacts = contact  # 分配上一帧足端接触缓存

        self._update_goals()  # 更新当前目标和下一目标，奖励与观测随后都会读取它们
        self._post_physics_step_callback()  # 刷新命令、地形 scan 和随机外推等 step 回调

        self.check_termination()  # 先判断终止，避免失败环境继续累计无效转移
        self.compute_reward()  # 根据最新状态计算本策略步奖励
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()  # 保存本次需要 reset 或更新的环境编号
        self.reset_idx(env_ids)  # 立即重置 done 环境，并把 episode 统计写入 extras

        self.cur_goals = self._gather_cur_goals()  # 缓存每个环境当前目标点
        self.next_goals = self._gather_cur_goals(future=1)  # 缓存每个环境下一目标点

        self.update_depth_buffer()  # 在需要时刷新视觉历史缓存

        self.compute_observations()  # 用最新状态拼接 actor 和 critic 输入

        self.last_actions[:] = self.actions[:]  # 保存上一策略步动作，用于动作平滑惩罚
        self.last_dof_vel[:] = self.dof_vel[:]  # 保存上一帧关节速度，用于关节加速度惩罚
        self.last_torques[:] = self.torques[:]  # 保存上一帧力矩，用于力矩变化惩罚
        self.last_root_vel[:] = self.root_states[:, 7:13]  # 保存上一帧根速度，用于估计机身加速度

        if self.viewer and self.enable_viewer_sync and self.debug_viz:  # viewer 回放时使用固定延迟，方便可视化对比
            self.gym.clear_lines(self.viewer)  # 清空上一帧调试绘制线段

            self._draw_goals()  # 在 viewer 中画出当前和未来目标点
            self._draw_feet()  # 在 viewer 中画出足端是否踩到边缘
            if self.cfg.depth.use_camera:  # 只在启用深度相机且到达刷新周期时执行
                window_name = "Depth Image"  # 统一深度图调试窗口名称，避免重复创建窗口
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 更新深度图调试窗口
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)  # 更新深度图调试窗口
                cv2.waitKey(1)  # 更新深度图调试窗口

    def reindex_feet(self, vec):  # 定义足端顺序重排逻辑
        return vec[:, [1, 0, 3, 2]]  # 返回足端顺序重排结果

    def reindex(self, vec):  # 定义动作顺序重排逻辑
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]  # 返回动作顺序重排结果

    def check_termination(self):  # 定义终止判断逻辑
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)  # 每步重新初始化 done 标记，再按失败条件逐项置位
        roll_cutoff = torch.abs(self.roll) > 1.5  # 标记横滚角过大的失败环境
        pitch_cutoff = torch.abs(self.pitch) > 1.5  # 标记俯仰角过大的失败环境
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals  # 标记已经完成全部目标的环境
        height_cutoff = self.root_states[:, 2] < -0.25  # 标记机身高度异常低的失败环境

        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # 标记超时或完成目标的 episode
        self.time_out_buf |= reach_goal_cutoff  # 标记超时或完成目标的 episode

        self.reset_buf |= self.time_out_buf  # 把超时或完成目标的环境加入 reset 集合
        self.reset_buf |= roll_cutoff  # 横滚过大时触发 reset
        self.reset_buf |= pitch_cutoff  # 俯仰过大时触发 reset
        self.reset_buf |= height_cutoff  # 机身高度异常时触发 reset

    def reset_idx(self, env_ids):  # 定义环境重置逻辑
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:  # 没有环境需要处理时跳过 reset 开销
            return  # 没有待处理环境时提前退出

        if self.cfg.terrain.curriculum:  # 只在课程学习启用时执行难度或命令更新
            self._update_terrain_curriculum(env_ids)  # 按上个 episode 表现调整这些环境的地形等级

        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):  # 只在课程学习启用时执行难度或命令更新
            self.update_command_curriculum(env_ids)  # 按课程进度扩展命令范围

        self._reset_dofs(env_ids)  # 先重置关节状态，避免旧姿态污染新 episode
        self._reset_root_states(env_ids)  # 再把 base 放回对应地形起点
        self._resample_commands(env_ids)  # 为新 episode 采样新的速度或 heading 命令
        self.gym.simulate(self.sim)  # 推进 PhysX 物理仿真一步
        self.gym.fetch_results(self.sim, True)  # 等待仿真完成，确保后续读取最新状态
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # 刷新刚体状态，用于足端位置和调试绘制

        self.last_actions[env_ids] = 0.  # 保存上一策略步动作，用于动作平滑惩罚
        self.last_dof_vel[env_ids] = 0.  # 保存上一帧关节速度，用于关节加速度惩罚
        self.last_torques[env_ids] = 0.  # 保存上一帧力矩，用于力矩变化惩罚
        self.last_root_vel[:] = 0.  # 保存上一帧根速度，用于估计机身加速度
        self.feet_air_time[env_ids] = 0.  # 累计足端离地时间
        self.reset_buf[env_ids] = 1  # 保存本步需要 reset 的环境布尔掩码
        self.obs_history_buf[env_ids, :, :] = 0.  # 分配历史本体观测缓存
        self.contact_buf[env_ids, :, :] = 0.  # 分配足端接触历史缓存
        self.action_history_buf[env_ids, :, :] = 0.  # 维护动作滑动窗口，用于模拟执行延迟
        self.cur_goal_idx[env_ids] = 0  # 保存每个环境当前 waypoint 索引
        self.reach_goal_timer[env_ids] = 0  # 累计停留在目标半径内的步数以做防抖

        self.extras["episode"] = {}  # 新建 episode 日志容器，runner 会从这里读取统计项
        for key in self.episode_sums.keys():  # 遍历所有启用奖励项，写出本 episode 的均值统计
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s  # 初始化 runner 读取的额外信息字典
            self.episode_sums[key][env_ids] = 0.  # 更新 episode 分项奖励统计
        self.episode_length_buf[env_ids] = 0  # 记录每个环境当前 episode 长度

        if self.cfg.terrain.curriculum:  # 只在课程学习启用时执行难度或命令更新
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())  # 初始化 runner 读取的额外信息字典
        if self.cfg.commands.curriculum:  # 只在课程学习启用时执行难度或命令更新
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]  # 初始化 runner 读取的额外信息字典

        if self.cfg.env.send_timeouts:  # 算法需要 timeout 标记时才把它写入 extras
            self.extras["time_outs"] = self.time_out_buf  # 初始化 runner 读取的额外信息字典

    def compute_reward(self):  # 定义奖励计算逻辑
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.  # 每步先清空总奖励，再累加各个奖励项
        for i in range(len(self.reward_functions)):  # 逐项遍历启用的奖励函数
            name = self.reward_names[i]  # 保存当前处理对象名称，便于查表或日志记录
            rew = self.reward_functions[i]() * self.reward_scales[name]  # 保存当前奖励项的批量值
            self.rew_buf += rew  # 把当前奖励项加入总奖励
            self.episode_sums[name] += rew  # 更新 episode 分项奖励统计
        if self.cfg.rewards.only_positive_rewards:  # 启用正奖励约束时才裁掉负总奖励
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)  # 裁掉负总奖励，降低早期探索时的惩罚冲击

        if "termination" in self.reward_scales:  # 终止奖励单独追加，避免被正奖励裁剪抹掉
            rew = self._reward_termination() * self.reward_scales["termination"]  # 保存当前奖励项的批量值
            self.rew_buf += rew  # 把当前奖励项加入总奖励
            self.episode_sums["termination"] += rew  # 更新 episode 分项奖励统计

    def compute_observations(self):  # 定义观测拼接逻辑
        """
        Computes observations
        """
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)  # 把 roll/pitch 组合成 IMU 姿态观测
        if self.global_counter % 5 == 0:  # 降低 yaw 误差刷新频率，减少重复三角计算
            self.delta_yaw = self.target_yaw - self.yaw  # 刷新当前目标方向相对机身 yaw 的误差
            self.delta_next_yaw = self.next_target_yaw - self.yaw  # 刷新下一目标方向相对机身 yaw 的误差
        obs_buf = torch.cat((  # 拼接当前帧 proprio 观测，后续同时进入 obs 和 history
                            self.base_ang_vel  * self.obs_scales.ang_vel,  # 加入缩放后的机身角速度观测
                            imu_obs,  # 加入 roll 和 pitch 组成的 IMU 姿态观测
                            0*self.delta_yaw[:, None],  # 保留旧输入槽位但屏蔽该 yaw 特征
                            self.delta_yaw[:, None],  # 加入当前目标方向相对机身 yaw 误差
                            self.delta_next_yaw[:, None],  # 加入下一目标方向误差，帮助策略提前转向
                            0*self.commands[:, 0:2],  # 保留历史命令槽位但屏蔽横向命令信息
                            self.commands[:, 0:1],  # 加入前向速度命令作为策略目标
                            (self.env_class != 17).float()[:, None],  # 加入非特定地形类别标志，帮助策略区分奖励分支
                            (self.env_class == 17).float()[:, None],  # 加入特定地形类别标志，帮助策略识别跳跃/障碍场景
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),  # 加入按策略顺序排列的关节位置偏差
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),  # 加入按策略顺序排列的关节速度
                            self.reindex(self.action_history_buf[:, -1]),  # 加入上一帧动作，让策略感知执行器历史
                            self.reindex_feet(self.contact_filt.float()-0.5),  # 加入足端接触状态，并按策略约定顺序排列
                            ),dim=-1)  # 完成本体观测拼接，形成单帧 proprio 向量
        priv_explicit = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,  # 保存显式特权观测片段
                                   0 * self.base_lin_vel,  # 保留显式特权槽位但不泄露额外线速度信息
                                   0 * self.base_lin_vel), dim=-1)  # 补齐显式特权观测维度，保持网络输入布局兼容
        priv_latent = torch.cat((  # 保存域随机化参数特权片段
            self.mass_params_tensor,  # 加入质量和质心随机化参数
            self.friction_coeffs_tensor,  # 加入地面摩擦随机化参数
            self.motor_strength[0] - 1,  # 加入 P 增益侧电机强度偏差
            self.motor_strength[1] - 1  # 加入 D 增益侧电机强度偏差
        ), dim=-1)  # 完成 privileged latent 拼接
        if self.cfg.terrain.measure_heights:  # 只在启用高度 scan 时采样或拼接地形高度
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)  # 保存机器人相对周围地形的高度 scan
            self.obs_buf = torch.cat([obs_buf, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)  # 保存 actor 网络最终观测
        else:  # 未启用质量随机化时记录零质量扰动
            self.obs_buf = torch.cat([obs_buf, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)  # 保存 actor 网络最终观测
        obs_buf[:, 6:8] = 0  # 写入历史前屏蔽 yaw 槽位，避免历史编码器直接依赖该角度
        self.obs_history_buf = torch.where(  # 按是否为首帧选择填满历史或滑动追加
            (self.episode_length_buf <= 1)[:, None, None],  # 识别 episode 起始帧，以决定历史缓存初始化方式
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),  # 首帧用当前观测复制填满 history，避免全零冷启动
            torch.cat([  # 非首帧时用滑动窗口追加最新历史帧
                self.obs_history_buf[:, 1:],  # 丢弃最旧观测，为当前帧腾出队尾位置
                obs_buf.unsqueeze(1)  # 把当前本体观测追加到历史队列末尾
            ], dim=1)  # 沿时间维拼接历史队列
        )  # 完成观测相关多行表达式的一部分

        self.contact_buf = torch.where(  # 按是否为首帧选择填满接触历史或滑动追加
            (self.episode_length_buf <= 1)[:, None, None],  # 识别 episode 起始帧，以决定历史缓存初始化方式
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),  # 首帧用当前接触状态填满 contact history
            torch.cat([  # 非首帧时用滑动窗口追加最新历史帧
                self.contact_buf[:, 1:],  # 丢弃最旧足端接触帧
                self.contact_filt.float().unsqueeze(1)  # 把当前足端接触帧追加到接触历史末尾
            ], dim=1)  # 沿时间维拼接历史队列
        )  # 完成观测相关多行表达式的一部分


    def get_noisy_measurement(self, x, scale):  # 定义传感噪声注入逻辑
        if self.cfg.noise.add_noise:  # 只有启用观测噪声时才扰动传感量
            x = x + (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level  # 按配置幅度加入均匀噪声，模拟传感器误差
        return x  # 返回传感噪声注入结果

    def create_sim(self):  # 定义仿真创建逻辑
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 指定 z 轴为竖直方向
        if self.cfg.depth.use_camera:  # 只在启用深度相机且到达刷新周期时执行
            depth_graphics_device_id = getattr(self.cfg.depth, "graphics_device_id", None)  # 允许命令行覆盖相机 graphics device
            self.graphics_device_id = self.sim_device_id if depth_graphics_device_id is None else depth_graphics_device_id  # 默认跟随仿真 GPU，也可手动指定
            print(f"Using camera graphics device id: {self.graphics_device_id}")  # 打印相机 graphics device，方便排查 CUDA/graphics 互操作问题
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)  # 保存 Isaac Gym 仿真实例
        mesh_type = self.cfg.terrain.mesh_type  # 读取地形网格类型，决定创建分支
        start = time()  # 记录创建地形开始时间
        print("*"*80)  # 输出创建或配置日志，便于定位运行阶段
        print("Start creating ground...")  # 输出创建或配置日志，便于定位运行阶段
        if mesh_type in ['heightfield', 'trimesh']:  # 高度场和三角网格都需要先实例化 Terrain 生成器
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)  # 实例化地形生成器，后续会取高度图和目标点
        if mesh_type=='plane':  # 平面地形走最轻量的 ground plane 创建路径
            self._create_ground_plane()  # 按 plane 配置创建无限平面地形
        elif mesh_type=='heightfield':  # 匹配另一种配置分支
            self._create_heightfield()  # 按 heightfield 配置创建高度场地形
        elif mesh_type=='trimesh':  # 匹配另一种配置分支
            self._create_trimesh()  # 按 trimesh 配置创建三角网格地形
        elif mesh_type is not None:  # 匹配另一种配置分支
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")  # 遇到非法配置时立即报错，避免继续产生错误状态
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))  # 输出创建或配置日志，便于定位运行阶段
        print("*"*80)  # 输出创建或配置日志，便于定位运行阶段
        self._create_envs()  # 地形就绪后创建机器人 actor 和并行环境

    def set_camera(self, position, lookat):  # 定义viewer 相机设置逻辑
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])  # 保存 viewer 相机位置
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])  # 保存 viewer 观察目标
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)  # 设置 viewer 相机观察位置和目标

    def _process_rigid_shape_props(self, props, env_id):  # 定义碰撞形状属性处理逻辑
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:  # 启用摩擦随机化时才为环境分配 friction bucket
            if env_id==0:  # 只在第一个环境初始化共享的属性缓存

                friction_range = self.cfg.domain_rand.friction_range  # 读取摩擦随机化范围
                num_buckets = 64  # 设置摩擦 bucket 数量以稳定随机分布
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))  # 为每个环境采样摩擦 bucket
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')  # 保存每个 bucket 的摩擦系数
                self.friction_coeffs = friction_buckets[bucket_ids]  # 保存每个环境实际使用的摩擦系数
            for s in range(len(props)):  # 遍历资产属性数组，把随机化结果写到每个形状或关节
                props[s].friction = self.friction_coeffs[env_id]  # 把随机化后的物理属性写回 Gym 属性对象
        return props  # 返回碰撞形状属性处理结果

    def _process_dof_props(self, props, env_id):  # 定义关节属性处理逻辑
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:  # 只在第一个环境初始化共享的属性缓存
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)  # 分配关节位置限制表，列 0/1 分别保存下界和上界
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # 读取 URDF 关节速度限制
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # 读取 URDF 关节力矩限制
            for i in range(len(props)):  # 遍历资产属性数组，把随机化结果写到每个形状或关节
                self.dof_pos_limits[i, 0] = props["lower"][i].item()  # 读取 URDF 关节位置下限
                self.dof_pos_limits[i, 1] = props["upper"][i].item()  # 读取 URDF 关节位置上限
                self.dof_vel_limits[i] = props["velocity"][i].item()  # 读取 URDF 关节速度限制
                self.torque_limits[i] = props["effort"][i].item()  # 读取 URDF 关节力矩限制

                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2  # 计算关节硬限位中心
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]  # 计算关节硬限位宽度
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit  # 把硬下限收缩成训练用 soft lower limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit  # 把硬上限收缩成训练用 soft upper limit
        return props  # 返回关节属性处理结果

    def _process_rigid_body_props(self, props, env_id):  # 定义刚体属性处理逻辑

        if self.cfg.domain_rand.randomize_base_mass:  # 启用质量随机化时才扰动 base 质量
            rng_mass = self.cfg.domain_rand.added_mass_range  # 读取 base 质量扰动范围
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))  # 采样 base 附加质量
            props[0].mass += rand_mass  # 把随机化后的物理属性写回 Gym 属性对象
        else:  # 未启用质心随机化时记录零质心扰动
            rand_mass = np.zeros((1, ))  # 采样 base 附加质量
        if self.cfg.domain_rand.randomize_base_com:  # 启用质心随机化时才扰动 base COM
            rng_com = self.cfg.domain_rand.added_com_range  # 读取质心扰动范围
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))  # 采样 base 质心偏移
            props[0].com += gymapi.Vec3(*rand_com)  # 把随机化后的物理属性写回 Gym 属性对象
        else:  # 未使用 heading 命令时直接采样 yaw 角速度命令
            rand_com = np.zeros(3)  # 采样 base 质心偏移
        mass_params = np.concatenate([rand_mass, rand_com])  # 合并质量和质心随机化参数
        return props, mass_params  # 返回刚体属性处理结果

    def _post_physics_step_callback(self):  # 定义物理步回调逻辑
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0)  # 标记到达命令重采样周期的环境
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())  # 给到达重采样周期的环境更新运动命令

        if self.cfg.commands.heading_command:  # 根据配置选择 heading 控制还是直接 yaw 速度控制
            forward = quat_apply(self.base_quat, self.forward_vec)  # 计算 base 前向量在世界系的方向
            heading = torch.atan2(forward[:, 1], forward[:, 0])  # 根据前向量得到当前 heading
            self.commands[:, 2] = torch.clip(0.8*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)  # 分配速度和 heading 命令缓存
            self.commands[:, 2] *= torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_clip  # 分配速度和 heading 命令缓存

        if self.cfg.terrain.measure_heights:  # 只在启用高度 scan 时采样或拼接地形高度
            if self.global_counter % self.cfg.depth.update_interval == 0:  # 只在启用深度相机且到达刷新周期时执行
                self.measured_heights = self._get_heights()  # 按刷新周期重新采样地形高度 scan
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):  # 到达外推间隔时才注入随机水平速度扰动
            self._push_robots()  # 按随机外推配置给机器人施加速度扰动

    def _gather_cur_goals(self, future=0):  # 定义目标读取逻辑
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)  # 返回目标读取结果

    def _resample_commands(self, env_ids):  # 定义命令重采样逻辑
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)  # 分配速度和 heading 命令缓存
        if self.cfg.commands.heading_command:  # 根据配置选择 heading 控制还是直接 yaw 速度控制
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)  # 分配速度和 heading 命令缓存
        else:  # 当前条件未命中时走默认处理路径
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)  # 分配速度和 heading 命令缓存
            self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_clip  # 分配速度和 heading 命令缓存

        self.commands[env_ids, :2] *= torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip  # 分配速度和 heading 命令缓存

    def _compute_torques(self, actions):  # 定义力矩计算逻辑
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """

        actions_scaled = actions * self.cfg.control.action_scale  # 把策略动作缩放到控制器物理尺度
        control_type = self.cfg.control.control_type  # 读取控制模式，决定动作如何解释
        if control_type=="P":  # 位置控制模式下把动作解释为关节位置偏移
            if not self.cfg.domain_rand.randomize_motor:  # 未随机化电机时使用标称 PD 增益
                torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains*self.dof_vel  # 保存控制器输出的关节力矩
            else:  # 电机随机化开启时把强度扰动乘到 P/D 项上
                torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel  # 保存控制器输出的关节力矩

        elif control_type=="V":  # 速度控制模式下把动作解释为目标关节速度
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt  # 保存控制器输出的关节力矩
        elif control_type=="T":  # 力矩控制模式下直接使用缩放后的动作
            torques = actions_scaled  # 保存控制器输出的关节力矩
        else:  # 未知控制模式无法安全映射到执行器力矩
            raise NameError(f"Unknown controller type: {control_type}")  # 遇到非法配置时立即报错，避免继续产生错误状态
        return torch.clip(torques, -self.torque_limits, self.torque_limits)  # 返回力矩计算结果

    def _reset_dofs(self, env_ids):  # 定义关节重置逻辑
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.9, (len(env_ids), self.num_dof), device=self.device)  # 从关节状态中切出位置视图
        self.dof_vel[env_ids] = 0.  # 从关节状态中切出速度视图

        env_ids_int32 = env_ids.to(dtype=torch.int32)  # 把环境编号转换成 Gym 索引接口要求的 int32
        self.gym.set_dof_state_tensor_indexed(self.sim,  # 把指定环境的关节状态写回仿真
                                              gymtorch.unwrap_tensor(self.dof_state),  # 传入要写回的关节状态和环境索引
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))  # 传入要写回的关节状态和环境索引
    def _reset_root_states(self, env_ids):  # 定义根状态重置逻辑
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """

        if self.custom_origins:  # 粗糙地形使用地形生成器给出的出生点
            self.root_states[env_ids] = self.base_init_state  # 包装根状态 buffer，按环境读取 base 状态
            self.root_states[env_ids, :3] += self.env_origins[env_ids]  # 包装根状态 buffer，按环境读取 base 状态
            if self.cfg.env.randomize_start_pos:  # 启用起点随机化时扰动 base 平面位置
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device)  # 包装根状态 buffer，按环境读取 base 状态
            if self.cfg.env.randomize_start_yaw:  # 启用朝向随机化时扰动初始 yaw
                rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)  # 采样初始 yaw 扰动
                if self.cfg.env.randomize_start_pitch:  # 启用 pitch 随机化时额外扰动初始俯仰角
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)  # 采样初始 pitch 扰动
                else:  # 当前条件未命中时走默认处理路径
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)  # 采样初始 pitch 扰动
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw)  # 把随机欧拉角转换为四元数
                self.root_states[env_ids, 3:7] = quat[:, :]  # 包装根状态 buffer，按环境读取 base 状态
            if self.cfg.env.randomize_start_y:  # 启用 y 方向随机化时沿横向移动出生点
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)  # 包装根状态 buffer，按环境读取 base 状态

        else:  # 当前条件未命中时走默认处理路径
            self.root_states[env_ids] = self.base_init_state  # 包装根状态 buffer，按环境读取 base 状态
            self.root_states[env_ids, :3] += self.env_origins[env_ids]  # 包装根状态 buffer，按环境读取 base 状态
        env_ids_int32 = env_ids.to(dtype=torch.int32)  # 把环境编号转换成 Gym 索引接口要求的 int32
        self.gym.set_actor_root_state_tensor_indexed(self.sim,  # 把指定环境的 root 状态写回仿真
                                                     gymtorch.unwrap_tensor(self.root_states),  # 传入要写回的 root 状态和环境索引
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))  # 传入要写回的 root 状态和环境索引

    def _push_robots(self):  # 定义随机外推逻辑
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy  # 读取随机外推允许的最大水平速度
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)  # 包装根状态 buffer，按环境读取 base 状态
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))  # 把所有环境的 root 状态写回仿真

    def _update_terrain_curriculum(self, env_ids):  # 定义地形课程更新逻辑
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """

        if not self.init_done:  # 初始化阶段不更新课程难度

            return  # 没有待处理环境时提前退出

        dis_to_origin = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)  # 计算机器人从出生点移动的距离
        threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s  # 按命令速度估计课程升级距离阈值
        move_up =dis_to_origin > 0.8*threshold  # 标记应升高地形难度的环境
        move_down = dis_to_origin < 0.4*threshold  # 标记应降低地形难度的环境

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down  # 采样每个环境初始地形等级

        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,  # 采样每个环境初始地形等级
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),  # 超过最高等级时随机回流，否则限制在合法等级内
                                                   torch.clip(self.terrain_levels[env_ids], 0))  # 超过最高等级时随机回流，否则限制在合法等级内
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]  # 分配环境出生点缓存
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]  # 分配地形类别缓存

        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]  # 取出当前地形 tile 对应的目标序列
        last_col = temp[:, -1].unsqueeze(1)  # 复制最后一个目标以补齐未来目标观测
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]  # 分配每个环境目标序列缓存
        self.cur_goals = self._gather_cur_goals()  # 缓存每个环境当前目标点
        self.next_goals = self._gather_cur_goals(future=1)  # 缓存每个环境下一目标点

    def _init_buffers(self):  # 定义缓存初始化逻辑
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # 获取 actor 根状态 GPU buffer
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # 获取关节状态 GPU buffer
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)  # 获取刚体接触力 GPU buffer
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)  # 获取足端力传感器 GPU buffer
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)  # 获取刚体状态 GPU buffer
        self.gym.refresh_dof_state_tensor(self.sim)  # 刷新关节位置和速度缓存
        self.gym.refresh_actor_root_state_tensor(self.sim)  # 刷新 base 根状态，用于位置、姿态和速度观测
        self.gym.refresh_net_contact_force_tensor(self.sim)  # 刷新接触力，用于碰撞惩罚和足端接触判断
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # 刷新刚体状态，用于足端位置和调试绘制
        self.gym.refresh_force_sensor_tensor(self.sim)  # 刷新足端力传感器读数

        self.root_states = gymtorch.wrap_tensor(actor_root_state)  # 包装根状态 buffer，按环境读取 base 状态
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)  # 包装刚体状态并整理成环境维度
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)  # 包装关节状态 buffer
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]  # 从关节状态中切出位置视图
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]  # 从关节状态中切出速度视图
        self.base_quat = self.root_states[:, 3:7]  # 缓存 base 姿态四元数

        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6)  # 整理四个足端的六维力传感器读数
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # 整理每个刚体 xyz 接触力

        self.common_step_counter = 0  # 记录全局公共步数，驱动命令和外推调度
        self.extras = {}  # 初始化 runner 读取的额外信息字典
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))  # 缓存世界系重力方向
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))  # 缓存 base 前向参考向量
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 分配力矩缓存，后续每个物理子步原地覆盖
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 分配比例增益缓存
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 分配微分增益缓存
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 保存裁剪后的动作，作为控制器输入
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 保存上一策略步动作，用于动作平滑惩罚
        self.last_dof_vel = torch.zeros_like(self.dof_vel)  # 保存上一帧关节速度，用于关节加速度惩罚
        self.last_torques = torch.zeros_like(self.torques)  # 保存上一帧力矩，用于力矩变化惩罚
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])  # 保存上一帧根速度，用于估计机身加速度

        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)  # 累计停留在目标半径内的步数以做防抖

        str_rng = self.cfg.domain_rand.motor_strength_range  # 读取电机强度随机化范围
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]  # 为 P/D 增益采样电机强度扰动
        if self.cfg.env.history_encoding:  # 只在启用历史编码时分配 proprio 历史缓存
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)  # 分配历史本体观测缓存
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs, device=self.device, dtype=torch.float)  # 维护动作滑动窗口，用于模拟执行延迟
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 4, device=self.device, dtype=torch.float)  # 分配足端接触历史缓存

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)  # 分配速度和 heading 命令缓存
        self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))  # 初始化所有环境的第一条运动命令
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)  # 保存命令观测缩放系数
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)  # 累计足端离地时间
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)  # 分配上一帧足端接触缓存
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # 缓存机身坐标系线速度
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 缓存机身坐标系角速度
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # 缓存重力在机身坐标系下的投影
        if self.cfg.terrain.measure_heights:  # 只在启用高度 scan 时采样或拼接地形高度
            self.height_points = self._init_height_points()  # 预生成高度采样点，后续 scan 只需旋转和平移
        self.measured_heights = 0  # 在首次 scan 前用标量占位，避免未初始化访问

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # 保存单环境默认关节角
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # 把默认关节角扩展到所有环境
        for i in range(self.num_dofs):  # 逐个关节匹配默认角度和增益
            name = self.dof_names[i]  # 保存当前处理对象名称，便于查表或日志记录
            angle = self.cfg.init_state.default_joint_angles[name]  # 读取当前关节默认角
            self.default_dof_pos[i] = angle  # 保存单环境默认关节角
            found = False  # 记录当前关节是否匹配到增益配置
            for dof_name in self.cfg.control.stiffness.keys():  # 遍历配置中的关节名片段以匹配 PD 增益
                if dof_name in name:  # 当前关节名匹配配置关键字时写入对应 PD 增益
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]  # 分配比例增益缓存
                    self.d_gains[i] = self.cfg.control.damping[dof_name]  # 分配微分增益缓存
                    found = True  # 记录当前关节是否匹配到增益配置
            if not found:  # 未匹配到增益配置时回退到零增益并提示
                self.p_gains[i] = 0.  # 分配比例增益缓存
                self.d_gains[i] = 0.  # 分配微分增益缓存
                if self.cfg.control.control_type in ["P", "V"]:  # 只有 PD 类控制模式缺增益时才打印警告
                    print(f"PD gain of joint {name} were not defined, setting them to zero")  # 输出创建或配置日志，便于定位运行阶段
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)  # 保存单环境默认关节角

        self.default_dof_pos_all[:] = self.default_dof_pos[0]  # 把默认关节角扩展到所有环境

        self.height_update_interval = 1  # 保存高度 scan 刷新间隔
        if hasattr(self.cfg.env, "height_update_dt"):  # 只有配置或状态存在时才启用兼容逻辑
            self.height_update_interval = int(self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))  # 保存高度 scan 刷新间隔

        if self.cfg.depth.use_camera:  # 只在启用深度相机且到达刷新周期时执行
            self.depth_buffer = torch.zeros(self.num_envs,  # 分配深度图历史缓存
                                            self.cfg.depth.buffer_len,  # 设置深度历史长度维度
                                            self.cfg.depth.resized[1],  # 补齐缓存张量的形状参数
                                            self.cfg.depth.resized[0]).to(self.device)  # 补齐缓存张量的形状参数

    def _prepare_reward_function(self):  # 定义奖励函数准备逻辑
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """

        for key in list(self.reward_scales.keys()):  # 遍历奖励权重配置
            scale = self.reward_scales[key]  # 读取当前奖励项权重
            if scale==0:  # 零权重奖励不加入运行时奖励列表
                self.reward_scales.pop(key)  # 删除零权重奖励，避免训练循环做无用调用
            else:  # 当前条件未命中时走默认处理路径
                self.reward_scales[key] *= self.dt  # 把奖励权重配置转成字典

        self.reward_functions = []  # 初始化奖励函数列表，后续按配置填充
        self.reward_names = []  # 初始化奖励名称列表，保证日志顺序和函数顺序一致
        for name, scale in self.reward_scales.items():  # 遍历奖励权重配置
            if name=="termination":  # termination 奖励单独处理，避免混入普通奖励循环
                continue  # 跳过 termination 项，后面会单独处理终止奖励
            self.reward_names.append(name)  # 保存奖励名称，便于日志和权重索引对齐
            name = '_reward_' + name  # 保存当前处理对象名称，便于查表或日志记录
            self.reward_functions.append(getattr(self, name))  # 把奖励名称解析成可直接调用的方法

        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)  # 更新 episode 分项奖励统计
                             for name in self.reward_scales.keys()}  # 遍历奖励权重配置

    def _create_ground_plane(self):  # 定义平面地形创建逻辑
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()  # 创建平面地形参数对象
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # 把配置参数写入地形创建对象
        plane_params.static_friction = self.cfg.terrain.static_friction  # 把配置参数写入地形创建对象
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # 把配置参数写入地形创建对象
        plane_params.restitution = self.cfg.terrain.restitution  # 把配置参数写入地形创建对象
        self.gym.add_ground(self.sim, plane_params)  # 把平面地形加入仿真

    def _create_heightfield(self):  # 定义高度场创建逻辑
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()  # 创建高度场参数对象
        hf_params.column_scale = self.cfg.terrain.horizontal_scale  # 把配置参数写入地形创建对象
        hf_params.row_scale = self.cfg.terrain.horizontal_scale  # 把配置参数写入地形创建对象
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale  # 把配置参数写入地形创建对象
        hf_params.nbRows = self.terrain.tot_cols  # 把配置参数写入地形创建对象
        hf_params.nbColumns = self.terrain.tot_rows  # 把配置参数写入地形创建对象
        hf_params.transform.p.x = -self.terrain.border  # 把配置参数写入地形创建对象
        hf_params.transform.p.y = -self.terrain.border  # 把配置参数写入地形创建对象
        hf_params.transform.p.z = 0.0  # 把配置参数写入地形创建对象
        hf_params.static_friction = self.cfg.terrain.static_friction  # 把配置参数写入地形创建对象
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # 把配置参数写入地形创建对象
        hf_params.restitution = self.cfg.terrain.restitution  # 把配置参数写入地形创建对象

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.flatten(order='C'), hf_params)  # 把高度场地形加入仿真
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)  # 初始化高度图缓存，地形创建后再填入真实数据

    def _create_trimesh(self):  # 定义三角网格地形创建逻辑
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()  # 创建三角网格参数对象
        tm_params.nb_vertices = self.terrain.vertices.shape[0]  # 把配置参数写入地形创建对象
        tm_params.nb_triangles = self.terrain.triangles.shape[0]  # 把配置参数写入地形创建对象

        tm_params.transform.p.x = -self.terrain.cfg.border_size  # 把配置参数写入地形创建对象
        tm_params.transform.p.y = -self.terrain.cfg.border_size  # 把配置参数写入地形创建对象
        tm_params.transform.p.z = 0.0  # 把配置参数写入地形创建对象
        tm_params.static_friction = self.cfg.terrain.static_friction  # 把配置参数写入地形创建对象
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # 把配置参数写入地形创建对象
        tm_params.restitution = self.cfg.terrain.restitution  # 把配置参数写入地形创建对象
        print("Adding trimesh to simulation...")  # 输出创建或配置日志，便于定位运行阶段
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  # 把三角网格地形加入仿真
        print("Trimesh added")  # 输出创建或配置日志，便于定位运行阶段
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)  # 初始化高度图缓存，地形创建后再填入真实数据
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)  # 缓存地形边缘掩码，用于足端踩边惩罚

    def attach_camera(self, i, env_handle, actor_handle):  # 定义相机挂载逻辑
        if self.cfg.depth.use_camera:  # 只在启用深度相机且到达刷新周期时执行
            config = self.cfg.depth  # 读取深度相机配置
            camera_props = gymapi.CameraProperties()  # 创建相机属性对象
            camera_props.width = self.cfg.depth.original[0]  # 把配置参数写入相机属性对象
            camera_props.height = self.cfg.depth.original[1]  # 把配置参数写入相机属性对象
            camera_props.enable_tensors = True  # 把配置参数写入相机属性对象
            camera_horizontal_fov = self.cfg.depth.horizontal_fov  # 读取相机水平视场角
            camera_props.horizontal_fov = camera_horizontal_fov  # 把配置参数写入相机属性对象

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)  # 保存新创建的相机句柄
            self.cam_handles.append(camera_handle)  # 保存相机句柄，后续刷新深度图时按环境读取

            local_transform = gymapi.Transform()  # 保存相机相对 base 的安装位姿

            camera_position = np.copy(config.position)  # 复制相机安装位置
            camera_angle = np.random.uniform(config.angle[0], config.angle[1])  # 采样相机俯仰角随机化

            local_transform.p = gymapi.Vec3(*camera_position)  # 设置相机相对 base link 的安装位姿
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)  # 设置相机相对 base link 的安装位姿
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)  # 获取 base link 句柄作为相机跟随目标

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)  # 把相机绑定到 base link 并随机器人运动

    def _create_envs(self):  # 定义并行环境创建逻辑
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)  # 将配置中的工程根目录占位符展开成真实资产路径
        asset_root = os.path.dirname(asset_path)  # Gym 加载接口需要目录和文件名分开传入
        asset_file = os.path.basename(asset_path)  # 只把 URDF/MJCF 文件名交给 load_asset

        asset_options = gymapi.AssetOptions()  # 为本次机器人资产导入收集所有物理和解析选项
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode  # 决定未单独设置的 DOF 默认使用哪种驱动模式
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints  # 合并固定关节可减少仿真刚体数量
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule  # 用胶囊体近似圆柱体以提升接触稳定性
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments  # 适配资产坐标系和 Isaac Gym 视觉附件约定
        asset_options.fix_base_link = self.cfg.asset.fix_base_link  # 需要固定基座实验时让 base 不参与动力学运动
        asset_options.density = self.cfg.asset.density  # 给缺少质量信息的几何体提供默认密度
        asset_options.angular_damping = self.cfg.asset.angular_damping  # 设置全局角阻尼，抑制不真实的高速旋转
        asset_options.linear_damping = self.cfg.asset.linear_damping  # 设置全局线阻尼，控制自由运动衰减
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity  # 限制刚体角速度，避免数值爆炸
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity  # 限制刚体线速度，避免异常碰撞造成发散
        asset_options.armature = self.cfg.asset.armature  # 给关节添加等效转子惯量，改善 PD 控制数值稳定性
        asset_options.thickness = self.cfg.asset.thickness  # 设置碰撞体厚度，影响薄几何体的接触鲁棒性
        asset_options.disable_gravity = self.cfg.asset.disable_gravity  # 支持调试或特殊实验中关闭机器人重力

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)  # 用上述选项加载一次资产，后续所有环境共享这份 asset
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)  # 记录 DOF 数量，后续按它分配动作、增益和关节缓存
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)  # 先读取刚体数量，供底层 buffer 形状初始化参考
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)  # 保留资产原始关节属性，创建每个 actor 前再按需处理
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)  # 保留原始碰撞属性，摩擦随机化会基于它改写

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)  # 通过名字查索引比依赖 URDF 顺序更稳
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)  # 关节名后续用于匹配默认角度和 PD 增益配置
        self.num_bodies = len(body_names)  # 用名称列表长度覆盖查询值，确保与索引收集逻辑一致
        self.num_dofs = len(self.dof_names)  # 缓存名称数量，后面遍历关节时避免重复查询 Gym
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]  # 按配置片段找足端，兼容不同机器人命名

        for s in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:  # 四足力传感器固定装在脚端，用于稳定读取接触力
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)  # 在 asset 层查找足端刚体，传感器必须在创建 actor 前挂上
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))  # 传感器放在足端局部原点，避免额外坐标偏移
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)  # 把传感器写入 asset，所有后续 actor 都会继承它

        penalized_contact_names = []  # 这些刚体接触不会终止 episode，但会进入碰撞惩罚
        for name in self.cfg.asset.penalize_contacts_on:  # 配置里使用名称片段，避免写死完整刚体名
            penalized_contact_names.extend([s for s in body_names if name in s])  # 展开成真实刚体名，后续再转 actor 内部索引
        termination_contact_names = []  # 这些刚体一旦接触地形就会触发失败 reset
        for name in self.cfg.asset.terminate_after_contacts_on:  # 终止接触也用片段匹配，方便跨资产复用配置
            termination_contact_names.extend([s for s in body_names if name in s])  # 收集完整名称，避免训练循环里反复做字符串匹配

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel  # 按 Isaac Gym root state 布局拼出 13 维初始状态
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)  # 放到训练设备上，reset 时可直接批量写入
        start_pose = gymapi.Transform()  # actor 创建接口需要 Transform，而不是 root state tensor
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])  # 先写入 base 初始平移，后面每个环境再叠加 origin

        self._get_env_origins()  # 在创建 actor 前确定每个环境应该落在哪个地形 tile 上
        env_lower = gymapi.Vec3(0., 0., 0.)  # 当前项目用 origin 控制摆放，环境包围盒本身不提供额外偏移
        env_upper = gymapi.Vec3(0., 0., 0.)  # lower/upper 保持零，让所有空间布局都由 terrain origin 管理
        self.actor_handles = []  # 保存每个 actor 句柄，后续查刚体索引和相机绑定都要用
        self.envs = []  # 保存每个环境句柄，后续 reset、渲染和传感器读取都要用
        self.cam_handles = []  # 深度相机按环境创建，句柄需要和 envs 一一对应
        self.cam_tensors = []  # 保留相机 tensor 容器，兼容已有视觉流程的接口预期
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)  # critic 需要知道每个环境采样到的质量和质心扰动

        print("Creating env...")  # 这一段创建大量 actor，打印阶段信息便于确认卡顿位置
        for i in tqdm(range(self.num_envs)):  # 每个并行环境都要单独应用随机化属性后再创建 actor

            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))  # 第四个参数控制 viewer 中环境网格排布密度
            pos = self.env_origins[i].clone()  # 从课程/地形结果复制出生点，避免随机扰动改写全局 origin
            if self.cfg.env.randomize_start_pos:  # 起点平移随机化让策略不只记住 tile 中心
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)  # 在当前 tile 内扰动 xy 出生位置
            if self.cfg.env.randomize_start_yaw:  # 初始朝向随机化让策略学会从不同朝向恢复目标跟踪
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.env.rand_yaw_range*np.random.uniform(-1, 1))  # Gym actor 创建阶段使用 gymapi.Quat 而不是 torch 四元数
                start_pose.r = rand_yaw_quat  # 本环境 actor 创建时使用随机 yaw 姿态
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))  # 把地形 origin 和配置中的 base 高度合成最终出生点

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)  # 为当前环境采样/写入摩擦等碰撞属性
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)  # Gym 只能在 create_actor 前把 shape 属性写回 asset
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)  # actor id 使用环境编号，便于 Gym 内部索引保持一致
            dof_props = self._process_dof_props(dof_props_asset, i)  # 首个环境会顺便缓存关节限位，返回值仍用于 actor 设置
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)  # 把关节驱动、限位等属性应用到刚创建的 actor
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)  # base 质量和质心随机化必须基于 actor 级刚体属性修改
            body_props, mass_params = self._process_rigid_body_props(body_props, i)  # 返回扰动后的属性，同时导出给 critic 的随机化参数
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)  # 改质量/质心后重算惯量，否则动力学参数不一致
            self.envs.append(env_handle)  # 后续访问相机、绘制 debug 和 reset 都按这个列表找环境
            self.actor_handles.append(anymal_handle)  # 后续从第一个 actor 查刚体索引，并在相机挂载时定位 base

            self.attach_camera(i, env_handle, anymal_handle)  # 如果启用深度输入，在 actor 创建后立即把相机绑到 base 上

            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)  # 把本环境随机化参数存成 tensor，训练时可直接拼进 privileged obs
        if self.cfg.domain_rand.randomize_friction:  # 只有摩擦随机化开启时 friction_coeffs 才会被前面的回调创建
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)  # 把每环境摩擦系数缓存到 GPU，供 privileged latent 使用

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)  # 按 actor 内部刚体索引缓存足端，接触奖励会高频读取
        for i in range(len(feet_names)):  # 用第一个环境查索引即可，所有同资产 actor 的刚体顺序一致
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])  # 将足端名称解析为可直接索引 contact buffer 的整数

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)  # 预先缓存惩罚接触索引，避免奖励函数里做字符串查找
        for i in range(len(penalized_contact_names)):  # 每个名称片段可能匹配多个刚体，需要逐个转成索引
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])  # 碰撞惩罚直接用这些索引切 contact_forces

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)  # 预先缓存终止接触索引，方便 reset 判断快速执行
        for i in range(len(termination_contact_names)):  # 和惩罚索引一样，只需在同资产的第一个 actor 上解析一次
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])  # 终止检查可以直接按这些刚体读接触力

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]  # 髋关节索引用于约束腿部外摆，名称顺序保持四足约定
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)  # 缓存成 tensor，奖励计算时可直接切 dof_pos
        for i, name in enumerate(hip_names):  # 用关节名解析索引，避免依赖 URDF 中 DOF 的排列
            self.hip_indices[i] = self.dof_names.index(name)  # 将髋关节名称转换成 dof_pos/dof_vel 的列索引
        thigh_names = ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]  # 大腿关节索引保留下来，便于后续奖励或调试扩展
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)  # 使用 tensor 缓存，和其他关节索引接口保持一致
        for i, name in enumerate(thigh_names):  # 逐个名称查找，保持前右/前左/后右/后左顺序
            self.thigh_indices[i] = self.dof_names.index(name)  # 将大腿关节名称转换成 DOF 列索引
        calf_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]  # 小腿关节索引用于腿部结构相关奖励或分析
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)  # 缓存小腿 DOF 索引，避免运行时重复查列表
        for i, name in enumerate(calf_names):  # 和髋/大腿保持同样的四足顺序
            self.calf_indices[i] = self.dof_names.index(name)  # 将小腿关节名称转换成 DOF 列索引

    def _get_env_origins(self):  # 定义环境原点计算逻辑
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:  # 高度场和三角网格都需要先实例化 Terrain 生成器
            self.custom_origins = True  # 标记是否使用地形生成器原点
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)  # 分配环境出生点缓存
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)  # 分配地形类别缓存

            max_init_level = self.cfg.terrain.max_init_terrain_level  # 确定初始地形难度上限
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1  # 只在课程学习启用时执行难度或命令更新
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)  # 采样每个环境初始地形等级
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)  # 为环境分配地形类型列
            self.max_terrain_level = self.cfg.terrain.num_rows  # 保存地形课程最大等级
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)  # 缓存地形平台中心
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]  # 分配环境出生点缓存

            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)  # 缓存地形类别表
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]  # 分配地形类别缓存

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)  # 缓存地形目标点表
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)  # 分配每个环境目标序列缓存
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)  # 保存每个环境当前 waypoint 索引
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]  # 取出当前地形 tile 对应的目标序列
            last_col = temp[:, -1].unsqueeze(1)  # 复制最后一个目标以补齐未来目标观测
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]  # 分配每个环境目标序列缓存
            self.cur_goals = self._gather_cur_goals()  # 缓存每个环境当前目标点
            self.next_goals = self._gather_cur_goals(future=1)  # 缓存每个环境下一目标点

        else:  # 当前条件未命中时走默认处理路径
            self.custom_origins = False  # 标记是否使用地形生成器原点
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)  # 分配环境出生点缓存

            num_cols = np.floor(np.sqrt(self.num_envs))  # 计算平地环境网格列数
            num_rows = np.ceil(self.num_envs / num_cols)  # 计算平地环境网格行数
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))  # 生成平地环境网格坐标
            spacing = self.cfg.env.env_spacing  # 读取平地环境间距
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]  # 分配环境出生点缓存
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]  # 分配环境出生点缓存
            self.env_origins[:, 2] = 0.  # 分配环境出生点缓存

    def _parse_cfg(self, cfg):  # 定义配置解析逻辑
        self.dt = self.cfg.control.decimation * self.sim_params.dt  # 计算策略控制周期
        self.obs_scales = self.cfg.normalization.obs_scales  # 缓存观测缩放配置
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)  # 把奖励权重配置转成字典
        reward_norm_factor = 1  # 保留奖励归一化入口
        for rew in self.reward_scales:  # 遍历奖励权重配置
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor  # 把奖励权重配置转成字典
        if self.cfg.commands.curriculum:  # 只在课程学习启用时执行难度或命令更新
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)  # 课程模式下使用可逐步扩展的命令范围
        else:  # 当前条件未命中时走默认处理路径
            self.command_ranges = class_to_dict(self.cfg.commands.max_ranges)  # 课程模式下使用可逐步扩展的命令范围
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:  # 非高度图地形不支持地形课程，直接关闭该开关
            self.cfg.terrain.curriculum = False  # 更新 cfg.terrain.curriculum 状态缓存
        self.max_episode_length_s = self.cfg.env.episode_length_s  # 保存 episode 秒级长度配置
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)  # 把 episode 秒数换算为策略步数

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)  # 更新 cfg.domain_rand.push_interval 状态缓存

    def _draw_height_samples(self):  # 定义高度点绘制逻辑
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """

        if not self.terrain.cfg.measure_heights:  # 只在启用高度 scan 时采样或拼接地形高度
            return  # 没有待处理环境时提前退出
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # 刷新刚体状态，用于足端位置和调试绘制
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))  # 创建调试线框球体
        i = self.lookat_id  # 保存循环索引
        base_pos = (self.root_states[i, :3]).cpu().numpy()  # 读取当前观察机器人的 base 位置
        heights = self.measured_heights[i].cpu().numpy()  # 保存机器人相对周围地形的高度 scan
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()  # 把高度采样点旋转到世界系
        for j in range(heights.shape[0]):  # 逐个高度采样点绘制调试球
            x = height_points[j, 0] + base_pos[0]  # 保存调试点 x 坐标
            y = height_points[j, 1] + base_pos[1]  # 保存调试点 y 坐标
            z = heights[j]  # 保存调试点 z 坐标
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)  # 创建调试几何体位姿
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)  # 把调试几何体绘制到 viewer 中

    def _draw_goals(self):  # 定义目标绘制逻辑
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))  # 创建调试线框球体
        sphere_geom_cur = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))  # 创建当前目标点绘制几何体
        sphere_geom_reached = gymutil.WireframeSphereGeometry(self.cfg.env.next_goal_threshold, 32, 32, None, color=(0, 1, 0))  # 创建到达半径绘制几何体
        goals = self.terrain_goals[self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]].cpu().numpy()  # 读取当前观察环境的目标序列
        for i, goal in enumerate(goals):  # 逐个绘制当前地形的目标点
            goal_xy = goal[:2] + self.terrain.cfg.border_size  # 把目标点平移到高度图坐标系
            pts = (goal_xy/self.terrain.cfg.horizontal_scale).astype(int)  # 把目标位置转换为高度图索引
            goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale  # 查询目标点地形高度
            pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)  # 创建调试绘制位姿
            if i == self.cur_goal_idx[self.lookat_id].cpu().item():  # 当前 goal 用特殊颜色绘制，便于检查目标切换
                gymutil.draw_lines(sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)  # 把调试几何体绘制到 viewer 中
                if self.reached_goal_ids[self.lookat_id]:  # 已到达目标时额外绘制到达半径
                    gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)  # 把调试几何体绘制到 viewer 中
            else:  # 当前条件未命中时走默认处理路径
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)  # 把调试几何体绘制到 viewer 中

        if not self.cfg.depth.use_camera:  # 只在启用深度相机且到达刷新周期时执行
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))  # 创建目标方向箭头几何体
            pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()  # 读取机器人位置作为箭头起点
            for i in range(5):  # 沿目标方向绘制一串箭头点
                norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)  # 计算目标向量长度，并为归一化提供分母
                target_vec_norm = self.target_pos_rel / (norm + 1e-5)  # 保存指向目标的单位向量
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()  # 计算箭头上的一个绘制点
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)  # 创建调试绘制位姿
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)  # 把调试几何体绘制到 viewer 中

            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))  # 创建目标方向箭头几何体
            for i in range(5):  # 沿目标方向绘制一串箭头点
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)  # 计算目标向量长度，并为归一化提供分母
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)  # 保存指向目标的单位向量
                pose_arrow = pose_robot[:2] + 0.2*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()  # 计算箭头上的一个绘制点
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)  # 创建调试绘制位姿
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)  # 把调试几何体绘制到 viewer 中

    def _draw_feet(self):  # 定义足端绘制逻辑
        if hasattr(self, 'feet_at_edge'):  # 只有配置或状态存在时才启用兼容逻辑
            non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))  # 创建非边缘足端几何体
            edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0, 0))  # 创建边缘足端几何体

            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]  # 读取足端世界坐标
            for i in range(4):  # 逐只脚绘制边缘接触状态
                pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1], feet_pos[self.lookat_id, i, 2]), r=None)  # 创建调试绘制位姿
                if self.feet_at_edge[self.lookat_id, i]:  # 只有足端踩边状态存在时才绘制该调试层
                    gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)  # 把调试几何体绘制到 viewer 中
                else:  # 当前条件未命中时走默认处理路径
                    gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)  # 把调试几何体绘制到 viewer 中

    def _init_height_points(self):  # 定义高度采样点初始化逻辑
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)  # 保存调试点 y 坐标
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)  # 保存调试点 x 坐标
        grid_x, grid_y = torch.meshgrid(x, y)  # 生成高度 scan 采样网格

        self.num_height_points = grid_x.numel()  # 保存高度采样点数量
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)  # 保存高度采样点或查询点
        for i in range(self.num_envs):  # 逐个处理并行环境
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()  # 采样高度点整体水平扰动
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset  # 采样高度点逐点水平噪声
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]  # 保存高度采样点或查询点
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]  # 保存高度采样点或查询点
        return points  # 返回高度采样点初始化结果

    def get_foot_contacts(self):  # 定义足端接触读取逻辑
        foot_contacts_bool = self.contact_forces[:, self.feet_indices, 2] > 10  # 根据足端竖直力得到接触状态
        if self.cfg.env.include_foot_contacts:  # 配置允许时才把足端接触暴露给策略
            return foot_contacts_bool  # 返回足端接触读取结果
        else:  # 当前条件未命中时走默认处理路径
            return torch.zeros_like(foot_contacts_bool).to(self.device)  # 返回足端接触读取结果

    def _get_heights(self, env_ids=None):  # 定义周围地形高度采样逻辑
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':  # 平面地形没有高度起伏，直接返回零 scan
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)  # 返回周围地形高度采样结果
        elif self.cfg.terrain.mesh_type == 'none':  # 匹配另一种配置分支
            raise NameError("Can't measure height with terrain mesh type 'none'")  # 遇到非法配置时立即报错，避免继续产生错误状态

        if env_ids:  # 传入环境子集时只采样这些环境以减少开销
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)  # 保存高度采样点或查询点
        else:  # 当前条件未命中时走默认处理路径
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)  # 保存高度采样点或查询点

        points += self.terrain.cfg.border_size  # 保存高度采样点或查询点
        points = (points/self.terrain.cfg.horizontal_scale).long()  # 保存高度采样点或查询点
        px = points[:, :, 0].view(-1)  # 保存高度图 x 索引
        py = points[:, :, 1].view(-1)  # 保存高度图 y 索引
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)  # 保存高度图 x 索引
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)  # 保存高度图 y 索引

        heights1 = self.height_samples[px, py]  # 读取当前格点高度
        heights2 = self.height_samples[px+1, py]  # 读取 x 相邻格点高度
        heights3 = self.height_samples[px, py+1]  # 读取 y 相邻格点高度
        heights = torch.min(heights1, heights2)  # 保存机器人相对周围地形的高度 scan
        heights = torch.min(heights, heights3)  # 保存机器人相对周围地形的高度 scan

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale  # 返回周围地形高度采样结果

    def _get_heights_points(self, coords, env_ids=None):  # 定义指定点高度查询逻辑
        if env_ids:  # 传入环境子集时只采样这些环境以减少开销
            points = coords[env_ids]  # 保存高度采样点或查询点
        else:  # 当前条件未命中时走默认处理路径
            points = coords  # 保存高度采样点或查询点

        points = (points/self.terrain.cfg.horizontal_scale).long()  # 保存高度采样点或查询点
        px = points[:, :, 0].view(-1)  # 保存高度图 x 索引
        py = points[:, :, 1].view(-1)  # 保存高度图 y 索引
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)  # 保存高度图 x 索引
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)  # 保存高度图 y 索引

        heights1 = self.height_samples[px, py]  # 读取当前格点高度
        heights2 = self.height_samples[px+1, py]  # 读取 x 相邻格点高度
        heights3 = self.height_samples[px, py+1]  # 读取 y 相邻格点高度
        heights = torch.min(heights1, heights2)  # 保存机器人相对周围地形的高度 scan
        heights = torch.min(heights, heights3)  # 保存机器人相对周围地形的高度 scan

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale  # 返回指定点高度查询结果

    def _reward_tracking_goal_vel(self):  # 定义目标速度奖励逻辑
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)  # 计算目标向量长度，并为归一化提供分母
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)  # 保存指向目标的单位向量
        cur_vel = self.root_states[:, 7:9]  # 读取机器人水平速度
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)  # 保存当前奖励项的批量值
        return rew  # 返回目标速度奖励结果

    def _reward_tracking_yaw(self):  # 定义朝向奖励逻辑
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))  # 保存当前奖励项的批量值
        return rew  # 返回朝向奖励结果

    def _reward_lin_vel_z(self):  # 定义竖直速度惩罚逻辑
        rew = torch.square(self.base_lin_vel[:, 2])  # 保存当前奖励项的批量值
        rew[self.env_class != 17] *= 0.5  # 保存 rew[self.env_class !，用于竖直速度惩罚
        return rew  # 返回竖直速度惩罚结果

    def _reward_ang_vel_xy(self):  # 定义横滚俯仰角速度惩罚逻辑
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)  # 返回横滚俯仰角速度惩罚结果

    def _reward_orientation(self):  # 定义姿态惩罚逻辑
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)  # 保存当前奖励项的批量值
        rew[self.env_class != 17] = 0.  # 保存 rew[self.env_class !，用于姿态惩罚
        return rew  # 返回姿态惩罚结果

    def _reward_dof_acc(self):  # 定义关节加速度惩罚逻辑
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)  # 返回关节加速度惩罚结果

    def _reward_collision(self):  # 定义碰撞惩罚逻辑
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)  # 返回碰撞惩罚结果

    def _reward_action_rate(self):  # 定义动作变化惩罚逻辑
        return torch.norm(self.last_actions - self.actions, dim=1)  # 返回动作变化惩罚结果

    def _reward_delta_torques(self):  # 定义力矩变化惩罚逻辑
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)  # 返回力矩变化惩罚结果

    def _reward_torques(self):  # 定义力矩大小惩罚逻辑
        return torch.sum(torch.square(self.torques), dim=1)  # 返回力矩大小惩罚结果

    def _reward_hip_pos(self):  # 定义髋关节姿态惩罚逻辑
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)  # 返回髋关节姿态惩罚结果

    def _reward_dof_error(self):  # 定义默认姿态偏差惩罚逻辑
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)  # 累计关节角偏离默认姿态的误差
        return dof_error  # 返回默认姿态偏差惩罚结果

    def _reward_feet_stumble(self):  # 定义足端绊倒惩罚逻辑

        rew = torch.any(  # 任一足端满足水平冲击过大时判为 stumble
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >  # 提取足端水平接触力大小，用来识别撞墙式接触
            4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)  # 水平力显著大于竖直支撑力时认为足端绊到障碍
        return rew.float()  # 返回足端绊倒惩罚结果

    def _reward_feet_edge(self):  # 定义足端踩边惩罚逻辑
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # 把足端位置转换为边缘掩码索引
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)  # 把足端位置转换为边缘掩码索引
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)  # 把足端位置转换为边缘掩码索引
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]  # 查询足端是否处在边缘区域

        self.feet_at_edge = self.contact_filt & feet_at_edge  # 缓存接触且踩边缘的足端状态
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)  # 保存当前奖励项的批量值
        return rew  # 返回足端踩边惩罚结果
