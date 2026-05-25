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

from posixpath import relpath  # 导入 posix 路径工具；当前文件未直接使用，可能保留自原始配置模板。
from torch.nn.modules.activation import ReLU  # 导入 ReLU 激活层；当前文件未直接使用，可能用于早期网络配置实验。
from torch.nn.modules.pooling import MaxPool2d  # 导入二维最大池化层；当前文件未直接使用，可能用于早期视觉网络配置实验。
from .base_config import BaseConfig  # 导入配置基类，用来把嵌套 class 配置递归实例化成可访问对象。
import torch.nn as nn  # 导入 PyTorch 神经网络模块别名；当前文件未直接使用，可能保留给网络结构配置扩展。
class LeggedRobotCfg(BaseConfig):  # 定义通用腿式机器人环境配置类，所有具体机器人环境配置通常继承它。
    class play:  # 定义播放/测试阶段相关配置。
        load_student_config = False  # 是否在 play 时加载 student/视觉策略配置，默认不加载。
        mask_priv_obs = False  # 是否在 play 时屏蔽 privileged observation，默认不屏蔽。
    class env:  # 定义环境规模、观测动作维度和 episode 行为等基础配置。
        num_envs = 6144  # 并行仿真的环境数量，也就是一次同时训练的机器人数量。

        n_scan = 132  # 地形扫描/高度点观测的维度，通常来自机器人周围的高度采样点。
        n_priv = 3+3 +3  # privileged state 维度，通常包含真实速度、外部扰动或地形等训练时可见信息。
        n_priv_latent = 4 + 1 + 12 +12  # privileged latent 维度，通常用于编码摩擦、质量、电机强度、关节偏差等隐变量。
        n_proprio = 3 + 2 + 3 + 4 + 36 + 5  # 本体感知观测维度，包含角速度、重力方向、命令、关节历史等 proprioceptive 信息。
        history_len = 10  # 历史本体感知观测长度，用于让策略从过去若干帧推断隐含状态。

        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12  # 策略输入 observation 总维度，由本体、扫描、历史和特权相关信息拼接得到。
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise  # privileged observation 维度；None 表示 critic 不额外使用单独的特权观测 buffer。
        num_actions = 12  # 策略动作维度，对四足机器人通常对应 12 个可控关节。
        env_spacing = 3.  # not used with heightfields/trimeshes  # 普通平面环境的间距；使用 heightfield/trimesh 地形时通常不会用到。
        send_timeouts = True # send time out information to the algorithm  # 是否把 episode 超时信息传给算法，便于正确处理时间截断。
        episode_length_s = 20   # 单个 episode 的最大持续时间，单位为秒。
        obs_type = "og"  # observation 类型标记，供环境内部选择不同观测组织方式。


        
        
        
        history_encoding = True  # 是否启用历史观测编码，用于 RMA/DAgger 相关历史 latent 学习。
        reorder_dofs = True  # 是否对关节自由度顺序重排，使策略动作/观测顺序与期望格式一致。
        
        
        # action_delay_range = [0, 5]

        # additional visual inputs 

        # action_delay_range = [0, 5]

        # additional visual inputs 
        include_foot_contacts = True  # 是否把足端接触状态纳入观测或辅助信息。
        
        randomize_start_pos = False  # 是否随机化机器人 episode 初始位置。
        randomize_start_vel = False  # 是否随机化机器人 episode 初始速度。
        randomize_start_yaw = False  # 是否随机化机器人 episode 初始 yaw 朝向。
        rand_yaw_range = 1.2  # 初始 yaw 随机范围，启用 randomize_start_yaw 时生效。
        randomize_start_y = False  # 是否随机化机器人初始 y 方向位置。
        rand_y_range = 0.5  # 初始 y 方向位置随机范围，启用 randomize_start_y 时生效。
        randomize_start_pitch = False  # 是否随机化机器人初始 pitch 姿态。
        rand_pitch_range = 1.6  # 初始 pitch 随机范围，启用 randomize_start_pitch 时生效。

        contact_buf_len = 100  # 接触历史 buffer 长度，用于记录足端或机体接触状态。

        next_goal_threshold = 0.2  # 判断是否到达当前 parkour goal 的距离阈值。
        reach_goal_delay = 0.1  # 到达 goal 后切换到下一个 goal 的延迟时间。
        num_future_goal_obs = 2  # observation 中包含的未来 goal 数量，用于引导机器人提前规划。

    class depth:  # 定义深度相机和视觉蒸馏相关配置。
        use_camera = False  # 是否启用深度相机输入；开启后通常进入视觉蒸馏/学生策略训练。
        camera_num_envs = 192  # 使用相机训练时的并行环境数量，通常小于非视觉训练以降低渲染开销。
        camera_terrain_num_rows = 10  # 使用相机训练时生成的地形行数。
        camera_terrain_num_cols = 20  # 使用相机训练时生成的地形列数。

        position = [0.27, 0, 0.03]  # front camera  # 深度相机相对于机器人机体的安装位置。
        angle = [-5, 5]  # positive pitch down  # 深度相机 pitch 角度范围，正值表示向下俯视。

        update_interval = 5  # 5 works without retraining, 8 worse  # 深度图更新间隔，表示每隔多少控制步刷新一次相机图像。

        original = (106, 60)  # 原始深度图分辨率，格式通常为宽高。
        resized = (87, 58)  # 网络输入使用的缩放后深度图分辨率，格式通常为宽高。
        horizontal_fov = 87  # 深度相机水平视场角，单位为度。
        buffer_len = 2  # 深度图历史帧 buffer 长度，用于给视觉网络提供时间信息。
        
        near_clip = 0  # 深度相机近裁剪距离，小于该距离的深度会被裁剪。
        far_clip = 2  # 深度相机远裁剪距离，大于该距离的深度会被裁剪。
        dis_noise = 0.0  # 深度测距噪声强度，默认不添加噪声。
        
        scale = 1  # 深度图缩放系数，用于归一化或调整输入尺度。
        invert = True  # 是否反转深度图数值方向，取决于后续网络期望的深度表示。

    class normalization:  # 定义观测和动作归一化/裁剪配置。
        class obs_scales:  # 定义不同观测项的缩放系数。
            lin_vel = 2.0  # 线速度观测缩放系数。
            ang_vel = 0.25  # 角速度观测缩放系数。
            dof_pos = 1.0  # 关节位置观测缩放系数。
            dof_vel = 0.05  # 关节速度观测缩放系数。
            height_measurements = 5.0  # 地形高度测量观测缩放系数。
        clip_observations = 100.  # observation 裁剪范围，防止异常大值进入策略网络。
        clip_actions = 1.2  # action 裁剪范围，限制策略输出幅度。
    class noise:  # 定义观测噪声配置。
        add_noise = False  # 是否给 observation 添加噪声，训练鲁棒策略时可开启。
        noise_level = 1.0 # scales other values  # 全局噪声强度倍率，会缩放各项噪声。
        quantize_height = True  # 是否对高度测量进行量化，模拟真实传感器分辨率限制。
        class noise_scales:  # 定义不同观测项的噪声尺度。
            rotation = 0.0  # 姿态/旋转相关观测噪声尺度。
            dof_pos = 0.01  # 关节位置观测噪声尺度。
            dof_vel = 0.05  # 关节速度观测噪声尺度。
            lin_vel = 0.05  # 线速度观测噪声尺度。
            ang_vel = 0.05  # 角速度观测噪声尺度。
            gravity = 0.02  # 重力方向投影观测噪声尺度。
            height_measurements = 0.02  # 地形高度测量噪声尺度。

    class terrain:  # 定义地形生成和地形 curriculum 相关配置。
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh  # 地形网格类型，trimesh 表示使用三角网格地形。
        hf2mesh_method = "grid"  # grid or fast  # heightfield 转 mesh 的方法，grid 更规则，fast 更快。
        max_error = 0.1 # for fast  # fast 转换方式允许的最大误差。
        max_error_camera = 2  # 相机训练时允许的地形简化最大误差，用于降低渲染和构建开销。

        y_range = [-0.4, 0.4]  # parkour goal 或可通行区域在 y 方向的随机范围。
        
        edge_width_thresh = 0.05  # 地形边缘宽度阈值，用于检测脚是否踩到边缘等。
        horizontal_scale = 0.05 # [m] influence computation time by a lot  # 地形水平分辨率，越小越精细但计算更慢。
        horizontal_scale_camera = 0.1  # 相机训练时使用的地形水平分辨率，通常更粗以提升效率。
        vertical_scale = 0.005 # [m]  # 地形垂直高度分辨率。
        border_size = 5 # [m]  # 地形外围边界尺寸，防止机器人走出有效区域。
        height = [0.02, 0.06]  # 随机粗糙地形的高度范围。
        simplify_grid = False  # 是否简化地形网格，开启后可降低复杂度。
        gap_size = [0.02, 0.1]  # 普通 gap 地形的缝隙尺寸范围。
        stepping_stone_distance = [0.02, 0.08]  # stepping stones 地形石块间距范围。
        downsampled_scale = 0.075  # 下采样地形尺度，用于生成或处理高度场。
        curriculum = True  # 是否启用地形 curriculum，根据机器人表现逐步提高地形难度。

        all_vertical = False  # 是否将地形障碍设置为完全竖直边界。
        no_flat = True  # 是否避免生成纯平地形，使训练更聚焦复杂 terrain。
        
        static_friction = 1.0  # 地形静摩擦系数。
        dynamic_friction = 1.0  # 地形动摩擦系数。
        restitution = 0.  # 地形碰撞恢复系数，0 表示无弹性反弹。
        measure_heights = True  # 是否采样机器人周围地形高度作为观测。
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)  # 高度采样点在机器人坐标系 x 方向的位置列表。
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]  # 高度采样点在机器人坐标系 y 方向的位置列表。
        measure_horizontal_noise = 0.0  # 高度采样点水平位置扰动强度，默认不添加。

        selected = False # select a unique terrain type and pass all arguments  # 是否只选择一种指定地形类型，而不是按比例混合地形。
        terrain_kwargs = None # Dict of arguments for selected terrain  # selected=True 时传给指定地形生成函数的参数字典。
        max_init_terrain_level = 5 # starting curriculum state  # 初始地形难度等级上限，用于 curriculum 初始化。
        terrain_length = 18.  # 单块地形在 x 方向的长度，单位米。
        terrain_width = 4  # 单块地形在 y 方向的宽度，单位米。
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !  # 地形行数，通常对应不同难度等级。
        num_cols = 40 # number of terrain cols (types)  # 地形列数，通常对应不同地形类型或随机实例。
        
        terrain_dict = {"smooth slope": 0.,  # 平滑斜坡地形采样权重。
                        "rough slope up": 0.0,  # 上坡粗糙斜坡地形采样权重。
                        "rough slope down": 0.0,  # 下坡粗糙斜坡地形采样权重。
                        "rough stairs up": 0.,  # 上楼梯粗糙地形采样权重。
                        "rough stairs down": 0.,  # 下楼梯粗糙地形采样权重。
                        "discrete": 0.,  # 离散随机障碍地形采样权重。
                        "stepping stones": 0.0,  # 踏石地形采样权重。
                        "gaps": 0.,  # 普通沟壑地形采样权重。
                        "smooth flat": 0,  # 平滑平地地形采样权重。
                        "pit": 0.0,  # 坑洞地形采样权重。
                        "wall": 0.0,  # 墙体障碍地形采样权重。
                        "platform": 0.,  # 平台地形采样权重。
                        "large stairs up": 0.,  # 大台阶上行地形采样权重。
                        "large stairs down": 0.,  # 大台阶下行地形采样权重。
                        "parkour": 0.2,  # 综合 parkour 地形采样权重。
                        "parkour_hurdle": 0.2,  # 跨栏 parkour 地形采样权重。
                        "parkour_flat": 0.2,  # 平台/平路 parkour 地形采样权重。
                        "parkour_step": 0.2,  # 台阶 parkour 地形采样权重。
                        "parkour_gap": 0.2,  # 跳沟/跨 gap parkour 地形采样权重。
                        "demo": 0.0,}  # demo 组合地形采样权重。
        terrain_proportions = list(terrain_dict.values())  # 将 terrain_dict 的权重提取成列表，供地形生成器计算累计比例。
        
        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces  # trimesh 中超过该坡度阈值的面会被修正为竖直面。
        origin_zero_z = True  # 是否让地形原点 z 坐标归零，方便机器人初始高度和 goal 计算。

        num_goals = 8  # 每条 parkour 路线上设置的目标点数量。

    class commands:  # 定义速度/航向命令和 goal 跟踪命令相关配置。
        curriculum = False  # 是否启用 command curriculum，逐步扩大命令范围。
        max_curriculum = 1.  # command curriculum 的最大进度或最大难度比例。
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)  # 命令维度，通常包含 x/y 线速度、yaw 角速度或 heading。
        resampling_time = 6. # time before command are changed[s]  # 命令重采样间隔，单位秒。
        heading_command = True # if true: compute ang vel command from heading error  # 是否使用 heading 命令，并由 heading 误差计算 yaw 角速度命令。
        
        lin_vel_clip = 0.2  # 线速度命令误差或观测的裁剪阈值。
        ang_vel_clip = 0.4  # 角速度命令误差或观测的裁剪阈值。
        # Easy ranges
        class ranges:  # 定义初始或简单阶段的命令采样范围。
            lin_vel_x = [0., 1.5] # min max [m/s]  # x 方向线速度命令范围。
            lin_vel_y = [0.0, 0.0]   # min max [m/s]  # y 方向线速度命令范围。
            ang_vel_yaw = [0, 0]    # min max [rad/s]  # yaw 角速度命令范围。
            heading = [0, 0]  # heading 目标角范围。

        # Easy ranges
        class max_ranges:  # 定义 curriculum 最终可达到的最大命令采样范围。
            lin_vel_x = [0.3, 0.8] # min max [m/s]  # curriculum 后 x 方向线速度命令范围。
            lin_vel_y = [-0.3, 0.3]#[0.15, 0.6]   # min max [m/s]  # curriculum 后 y 方向线速度命令范围。
            ang_vel_yaw = [-0, 0]    # min max [rad/s]  # curriculum 后 yaw 角速度命令范围。
            heading = [-1.6, 1.6]  # curriculum 后 heading 目标角范围。

        class crclm_incremnt:  # 定义 command curriculum 每次扩大范围的增量。
            lin_vel_x = 0.1 # min max [m/s]  # x 方向速度范围增量。
            lin_vel_y = 0.1  # min max [m/s]  # y 方向速度范围增量。
            ang_vel_yaw = 0.1    # min max [rad/s]  # yaw 角速度范围增量。
            heading = 0.5  # heading 范围增量。

        waypoint_delta = 0.7  # parkour waypoint/goal 之间期望的前进距离或目标间隔。

    class init_state:  # 定义机器人初始根状态和默认关节角。
        pos = [0.0, 0.0, 1.] # x,y,z [m]  # 机器人 base 初始位置。
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]  # 机器人 base 初始姿态四元数。
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]  # 机器人 base 初始线速度。
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]  # 机器人 base 初始角速度。
        default_joint_angles = { # target angles when action = 0.0  # action 为 0 时各关节对应的默认目标角。
            "joint_a": 0.,  # 示例关节 joint_a 的默认角度，具体机器人配置会覆盖。
            "joint_b": 0.}  # 示例关节 joint_b 的默认角度，具体机器人配置会覆盖。

    class control:  # 定义动作到关节控制目标/力矩的转换参数。
        control_type = 'P' # P: position, V: velocity, T: torques  # 控制模式，P 表示位置 PD 控制，V 表示速度控制，T 表示直接力矩控制。
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]  # 关节位置 PD 刚度字典，具体机器人配置会按关节名覆盖。
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]  # 关节速度阻尼字典，具体机器人配置会按关节名覆盖。
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5  # 策略 action 到目标关节角偏移的缩放系数。
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 每个策略 step 内执行的物理仿真子步数，也就是控制降采样比例。

    class asset:  # 定义机器人资产加载和刚体属性配置。
        file = ""  # 机器人 URDF/MJCF 资产文件路径，具体机器人配置必须覆盖。
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors  # 足端刚体名称关键字，用于从刚体列表中找到脚。
        penalize_contacts_on = []  # 与这些刚体发生接触时施加奖励惩罚。
        terminate_after_contacts_on = []  # 与这些刚体发生接触时终止 episode。
        disable_gravity = False  # 是否关闭机器人资产的重力。
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">  # 是否合并固定关节连接的刚体以提升仿真效率。
        fix_base_link = False # fixe the base of the robot  # 是否固定机器人 base link，通常训练运动策略时为 False。
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)  # 默认关节驱动模式，3 表示 effort/力矩模式。
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter  # 自碰撞开关位，0 通常表示启用自碰撞，1 表示禁用。
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation  # 是否用 capsule 替代 cylinder 碰撞体以提高仿真稳定性。
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up  # 是否翻转视觉 mesh 坐标，使 y-up 资产适配 z-up 仿真坐标。
        
        density = 0.001  # 资产默认密度，用于没有显式质量属性的刚体。
        angular_damping = 0.  # 刚体角阻尼。
        linear_damping = 0.  # 刚体线阻尼。
        max_angular_velocity = 1000.  # 刚体最大角速度限制。
        max_linear_velocity = 1000.  # 刚体最大线速度限制。
        armature = 0.  # 关节 armature 惯性参数，用于提高仿真数值稳定性。
        thickness = 0.01  # 碰撞体厚度参数。

    class domain_rand:  # 定义 domain randomization 参数，用于提升 sim-to-real 鲁棒性。
        randomize_friction = True  # 是否随机化地面/接触摩擦系数。
        friction_range = [0.6, 2.]  # 摩擦系数随机采样范围。
        randomize_base_mass = True  # 是否随机化机器人 base 质量。
        added_mass_range = [0., 3.]  # base 额外质量随机范围，单位通常为 kg。
        randomize_base_com = True  # 是否随机化机器人 base 质心位置。
        added_com_range = [-0.2, 0.2]  # base 质心偏移随机范围。
        push_robots = True  # 是否周期性随机推动机器人。
        push_interval_s = 8  # 随机 push 的时间间隔，单位秒。
        max_push_vel_xy = 0.5  # 随机 push 施加的最大 xy 平面速度变化。

        randomize_motor = True  # 是否随机化电机强度。
        motor_strength_range = [0.8, 1.2]  # 电机强度倍率随机范围。

        delay_update_global_steps = 24 * 8000  # 动作延迟 curriculum 更新间隔，按全局环境步计。
        action_delay = False  # 是否启用动作延迟，用于模拟控制链路延迟。
        action_curr_step = [1, 1]  # 当前动作延迟步数调度列表。
        action_curr_step_scratch = [0, 1]  # 从零训练时使用的动作延迟调度列表。
        action_delay_view = 1  # viewer/play 模式下使用的动作延迟步数。
        action_buf_len = 8  # 动作历史 buffer 长度，用于实现动作延迟。
        
    class rewards:  # 定义奖励函数权重和奖励裁剪/限制参数。
        class scales:  # 定义各个奖励项的权重，正数鼓励，负数惩罚。
            # tracking rewards
            tracking_goal_vel = 1.5  # goal 方向速度跟踪奖励权重，鼓励朝目标移动。
            tracking_yaw = 0.5  # yaw/朝向跟踪奖励权重，鼓励朝向目标方向。
            # regularization rewards
            lin_vel_z = -1.0  # base 垂直速度惩罚权重，抑制上下跳动。
            ang_vel_xy = -0.05  # base roll/pitch 角速度惩罚权重，抑制机身晃动。
            orientation = -1.  # 姿态偏离惩罚权重，鼓励机身保持合理朝向。
            dof_acc = -2.5e-7  # 关节加速度惩罚权重，鼓励动作平滑。
            collision = -10.  # 非期望碰撞惩罚权重，例如大腿/机身碰撞地面。
            action_rate = -0.1  # action 变化率惩罚权重，鼓励策略输出平滑。
            delta_torques = -1.0e-7  # 力矩变化惩罚权重，鼓励执行器输出平滑。
            torques = -0.00001  # 力矩大小惩罚权重，鼓励节能。
            hip_pos = -0.5  # 髋关节位置偏离惩罚权重，约束腿部姿态。
            dof_error = -0.04  # 关节角偏离默认姿态的惩罚权重。
            feet_stumble = -1  # 脚碰到垂直边缘或绊倒相关惩罚权重。
            feet_edge = -1  # 脚踩地形边缘相关惩罚权重。
            
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)  # 在训练早期，如果reward为负数，则裁剪为零。
        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)  # 跟踪类指数奖励的 sigma 参数，控制误差敏感度。
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized  # 软关节位置限制比例，超过 URDF 限制该比例后惩罚。
        soft_dof_vel_limit = 1  # 软关节速度限制比例，超过限制后惩罚。
        soft_torque_limit = 0.4  # 软力矩限制比例，超过限制后惩罚。
        base_height_target = 1.  # 期望 base 高度，具体机器人配置通常会覆盖。
        max_contact_force = 40. # forces above this value are penalized  # 最大允许接触力，超过该值会产生惩罚。





    # viewer camera:
    class viewer:  # 定义 Isaac Gym viewer 默认相机配置。
        ref_env = 0  # viewer 默认参考的环境编号。
        pos = [10, 0, 6]  # [m]  # viewer 相机默认位置。
        lookat = [11., 5, 3.]  # [m]  # viewer 相机默认观察目标点。

    class sim:  # 定义 Isaac Gym 仿真全局参数。
        dt =  0.005  # 仿真时间步长，单位秒。
        substeps = 1  # 每个仿真步内部的子步数量。
        gravity = [0., 0. ,-9.81]  # [m/s^2]  # 重力加速度向量。
        up_axis = 1  # 0 is y, 1 is z  # 仿真世界上方向轴，1 表示 z 轴向上。

        class physx:  # 定义 PhysX 物理引擎参数。
            num_threads = 10  # PhysX 使用的 CPU 线程数。
            solver_type = 1  # 0: pgs, 1: tgs  # PhysX 求解器类型，1 表示 TGS。
            num_position_iterations = 4  # 位置约束求解迭代次数。
            num_velocity_iterations = 0  # 速度约束求解迭代次数。
            contact_offset = 0.01  # [m]  # 接触检测偏移距离，影响碰撞提前检测。
            rest_offset = 0.0   # [m]  # 静止接触偏移距离。
            bounce_threshold_velocity = 0.5 #0.5 [m/s]  # 低于该速度的碰撞不产生明显反弹。
            max_depenetration_velocity = 1.0  # 物体穿透修正时允许的最大分离速度。
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more  # GPU 上最大接触对数量，环境数多时需要更大 buffer。
            default_buffer_size_multiplier = 5  # PhysX 默认 buffer 大小倍率，用于容纳更多接触和约束数据。
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)  # 接触数据收集模式，2 表示收集所有子步接触。

class LeggedRobotCfgPPO(BaseConfig):  # 定义通用腿式机器人 PPO 训练配置类，具体机器人训练配置通常继承它。
    seed = 1  # 随机种子，用于控制环境、网络初始化和采样的可复现性。
    runner_class_name = 'OnPolicyRunner'  # 训练 runner 类名，task_registry 会用它对应 rsl_rl 中的 on-policy runner。
 
    class policy:  # 定义策略网络和 actor-critic 结构参数。
        init_noise_std = 1.0  # 初始动作高斯分布标准差，用于 PPO 探索。
        continue_from_last_std = True  # resume 时是否沿用 checkpoint 中的动作标准差。
        scan_encoder_dims = [128, 64, 32]  # 地形 scan/height 输入编码器的隐藏层维度。
        actor_hidden_dims = [512, 256, 128]  # actor MLP 隐藏层维度。
        critic_hidden_dims = [512, 256, 128]  # critic MLP 隐藏层维度。
        priv_encoder_dims = [64, 20]  # privileged latent 编码器隐藏层维度。
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid  # 网络激活函数类型。
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'  # 循环网络类型，仅在使用 recurrent actor-critic 时生效。
        rnn_hidden_size = 512  # RNN 隐状态维度。
        rnn_num_layers = 1  # RNN 层数。

        tanh_encoder_output = False  # 是否对 encoder 输出使用 tanh 限幅。
    
    class algorithm:  # 定义 PPO 和相关辅助训练损失的超参数。
        # training params
        value_loss_coef = 1.0  # critic value loss 权重。
        use_clipped_value_loss = True  # 是否使用 PPO clipped value loss。
        clip_param = 0.2  # PPO policy ratio 裁剪阈值 epsilon。
        entropy_coef = 0.01  # entropy bonus 权重，用于鼓励探索。
        num_learning_epochs = 5  # 每轮 rollout 后重复优化的 epoch 数。
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches  # 每个 PPO update 切分出的 mini-batch 数量。
        learning_rate = 2.e-4 #5.e-4  # PPO optimizer 学习率。
        schedule = 'adaptive' # could be adaptive, fixed  # 学习率调度方式，adaptive 通常根据 KL 调整。
        gamma = 0.99  # 折扣因子，控制未来奖励权重。
        lam = 0.95  # GAE lambda 参数，控制 bias-variance tradeoff。
        desired_kl = 0.01  # 目标 KL 散度，用于 adaptive 学习率调节。
        max_grad_norm = 1.  # 梯度裁剪最大范数。
        # dagger params
        dagger_update_freq = 20  # DAgger/历史编码更新频率。
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]  # privileged regularization 系数调度，用于从特权 latent 过渡到历史 latent。
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]  # resume 阶段使用的 privileged regularization 系数调度。
    
    class depth_encoder:  # 定义深度相机 encoder 和视觉蒸馏训练参数。
        if_depth = LeggedRobotCfg.depth.use_camera  # 是否启用深度 encoder，默认跟随环境 depth.use_camera。
        depth_shape = LeggedRobotCfg.depth.resized  # 深度图输入分辨率，来自环境 depth.resized。
        buffer_len = LeggedRobotCfg.depth.buffer_len  # 深度图历史帧数，来自环境 depth.buffer_len。
        hidden_dims = 512  # 深度 encoder 隐层维度。
        learning_rate = 1.e-3  # 深度 encoder/视觉学生部分学习率。
        num_steps_per_env = LeggedRobotCfg.depth.update_interval * 24  # 视觉训练每个环境采集的步数，与深度更新间隔和 PPO rollout 长度相关。

    class estimator:  # 定义 privileged state estimator 的训练参数。
        train_with_estimated_states = True  # 是否使用估计器预测的隐状态参与训练。
        learning_rate = 1.e-4  # estimator optimizer 学习率。
        hidden_dims = [128, 64]  # estimator MLP 隐藏层维度。
        priv_states_dim = LeggedRobotCfg.env.n_priv  # estimator 输出的 privileged state 维度。
        num_prop = LeggedRobotCfg.env.n_proprio  # estimator 输入中的本体感知维度。
        num_scan = LeggedRobotCfg.env.n_scan  # estimator 输入中的地形扫描维度。

    class runner:  # 定义训练 runner 的运行、保存和恢复参数。
        policy_class_name = 'ActorCritic'  # 策略类名配置，当前 runner 实际构造 ActorCriticRMA。
        algorithm_class_name = 'PPO'  # 算法类名，runner 会用该名称创建 PPO 算法对象。
        num_steps_per_env = 24 # per iteration  # 每个 PPO iteration 中每个并行环境采集的 rollout 步数。
        max_iterations = 50000 # number of policy updates  # 最大策略更新迭代次数。

        # logging
        save_interval = 100 # check for potential saves every this many iterations  # checkpoint 保存检查间隔。
        experiment_name = 'rough_a1'  # 默认实验名称，会影响日志目录。
        run_name = ''  # 默认运行名称，可用于区分同一实验下不同 run。
        # load and resume
        resume = False  # 是否从已有 checkpoint 恢复训练。
        load_run = -1 # -1 = last run  # 要加载的 run 名称或编号，-1 表示自动选择最新 run。
        checkpoint = -1 # -1 = last saved model  # 要加载的 checkpoint 编号，-1 表示自动选择最新模型。
        resume_path = None # updated from load_run and chkpt  # 显式恢复路径，通常由 load_run 和 checkpoint 推导得到。
