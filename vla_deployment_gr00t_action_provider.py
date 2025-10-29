#!/usr/bin/env python3
"""
GR00T VLA Action Provider for Isaac Sim Deployment

这个模块一种ActionProvider，使用Isaac-GR00T模型进行推理，并将结果转换为机器人控制命令。
它实现了控制循环：获取图像 -> GR00T推理 -> 执行第一步动作 -> 循环
"""

import sys
import os
import time
from typing import Optional, Dict, Any
import numpy as np
import torch
from pathlib import Path

# 添加路径以便导入
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(project_root, 'unitree_sim_isaaclab'))

from action_provider.action_base import ActionProvider
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.video import VideoToTensor, VideoResize, VideoToNumpy
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.transforms import GR00TTransform


class CustomDataConfig:
    """自定义数据配置，匹配 checkpoint-59900 的 metadata"""
    
    def __init__(self):
        # 从 metadata.json 获取的配置
        self.video_keys = ["video.rs_view"]
        self.state_keys = ["state.left_arm", "state.right_arm", "state.left_ee", "state.right_ee"]
        self.action_keys = ["action.left_arm", "action.right_arm", "action.left_ee", "action.right_ee"]
        self.language_keys = []  # 这个 checkpoint 可能不使用语言
        
        # 观察索引（当前帧）
        self.observation_indices = [0]
        # 动作索引（预测未来16步）
        self.action_indices = list(range(16))
    
    def modality_config(self) -> Dict[str, ModalityConfig]:
        """创建模态配置"""
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
        }
    
    def transform(self) -> ComposedModalityTransform:
        """创建变换管道"""
        transforms = [
            # 视频变换流程
            VideoToTensor(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=224, width=224),
            VideoToNumpy(apply_to=self.video_keys),
            # 状态变换
            StateActionToTensor(apply_to=self.state_keys),
            # 状态归一化
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys}
            ),
            # 连接变换
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # GR00T 模型特定的变换
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class GR00TVLAActionProvider(ActionProvider):
    """使用GR00T VLA模型进行推理的ActionProvider"""
    
    def __init__(self, env, checkpoint_path: str = "checkpoint-59900", device: str = None, denoising_steps: int = 4):
        """
        初始化GR00T VLA ActionProvider
        
        Args:
            env: Isaac Sim环境实例
            checkpoint_path: GR00T模型检查点路径
            device: 计算设备 ("cuda" 或 "cpu")
            denoising_steps: 去噪步数
        """
        super().__init__("GR00TVLAActionProvider")
        self.env = env
        
        # 设置设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"[{self.name}] 使用设备: {self.device}")
        print(f"[{self.name}] 加载 checkpoint: {checkpoint_path}")
        
        # 创建数据配置
        self.data_config = CustomDataConfig()
        modality_config = self.data_config.modality_config()
        modality_transform = self.data_config.transform()
        
        # 加载模型
        print(f"[{self.name}] 正在加载GR00T模型...")
        try:
            self.policy = Gr00tPolicy(
                model_path=checkpoint_path,
                modality_config=modality_config,
                modality_transform=modality_transform,
                embodiment_tag="new_embodiment",
                denoising_steps=denoising_steps,
                device=self.device,
            )
            print(f"[{self.name}] 模型加载完成!")
        except Exception as e:
            print(f"[{self.name}] 模型加载失败: {e}")
            raise
        
        # 初始化关节映射
        self._setup_joint_mapping()
        
        # 当前动作缓存（用于存储完整的16步动作序列）
        self.current_action_sequence = None
        self.action_step_index = 0  # 当前执行的动作步索引
        
        # 控制参数
        self.max_action_steps = 16  # GR00T输出的动作序列长度
        self.action_execution_counter = 0  # 动作执行计数器
        
        print(f"[{self.name}] GR00T VLA ActionProvider 初始化完成")
    
    def _setup_joint_mapping(self):
        """设置机器人关节映射"""
        # 获取所有关节名称
        self.all_joint_names = self.env.scene["robot"].data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}
        
        # 左臂关节名称（7个）
        left_arm_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]
        
        # 右臂关节名称（7个）
        right_arm_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        
        # Inspire手部关节名称（12个）
        # 顺序：R_pinky, R_ring, R_middle, R_index, R_thumb_pitch, R_thumb_yaw,
        #        L_pinky, L_ring, L_middle, L_index, L_thumb_pitch, L_thumb_yaw
        inspire_hand_names = [
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_middle_proximal_joint",
            "R_index_proximal_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_proximal_yaw_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_middle_proximal_joint",
            "L_index_proximal_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_proximal_yaw_joint",
        ]
        
        # 获取关节索引
        self.left_arm_indices = [self.joint_to_index[name] for name in left_arm_names if name in self.joint_to_index]
        self.right_arm_indices = [self.joint_to_index[name] for name in right_arm_names if name in self.joint_to_index]
        self.inspire_hand_indices = [self.joint_to_index[name] for name in inspire_hand_names if name in self.joint_to_index]
        
        print(f"[{self.name}] 关节映射:")
        print(f"  左臂关节数: {len(self.left_arm_indices)}")
        print(f"  右臂关节数: {len(self.right_arm_indices)}")
        print(f"  Inspire手部关节数: {len(self.inspire_hand_indices)}")
    
    def _get_robot_state(self) -> Dict[str, np.ndarray]:
        """从环境获取机器人状态"""
        joint_pos = self.env.scene["robot"].data.joint_pos[0]  # [num_joints]
        
        # 提取左臂状态（7个关节）
        left_arm_state = joint_pos[self.left_arm_indices].cpu().numpy() if len(self.left_arm_indices) == 7 else np.zeros(7)
        
        # 提取右臂状态（7个关节）
        right_arm_state = joint_pos[self.right_arm_indices].cpu().numpy() if len(self.right_arm_indices) == 7 else np.zeros(7)
        
        # 提取Inspire手部状态
        # GR00T期望的left_ee和right_ee各6个值
        # 我们将12个inspire关节分成左右各6个
        if len(self.inspire_hand_indices) >= 12:
            inspire_positions = joint_pos[self.inspire_hand_indices].cpu().numpy()
            # 右手指6个关节，左手指6个关节
            right_ee_state = inspire_positions[:6] if len(inspire_positions) >= 6 else np.zeros(6)
            left_ee_state = inspire_positions[6:12] if len(inspire_positions) >= 12 else np.zeros(6)
        else:
            left_ee_state = np.zeros(6)
            right_ee_state = np.zeros(6)
        
        return {
            "left_arm": left_arm_state,
            "right_arm": right_arm_state,
            "left_ee": left_ee_state,
            "right_ee": right_ee_state,
        }
    
    def _get_camera_image(self) -> Optional[np.ndarray]:
        """从环境获取头部相机图像"""
        try:
            if "front_camera" in self.env.scene.keys():
                # 获取RGB图像 [batch, height, width, 3]
                image_tensor = self.env.scene["front_camera"].data.output["rgb"][0]
                # 转换为numpy数组 [height, width, 3]
                image = image_tensor.cpu().numpy()
                # 转换为uint8格式（GR00T期望RGB）
                if image.dtype != np.uint8:
                    # 如果是浮点型，假设范围在[0, 1]，转换为[0, 255]
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                # Isaac Sim通常输出RGB格式，直接使用
                # 添加时间维度 [1, height, width, 3]
                image = np.expand_dims(image, axis=0)
                return image
            else:
                print(f"[{self.name}] 警告: 未找到front_camera")
                return None
        except Exception as e:
            print(f"[{self.name}] 获取相机图像失败: {e}")
            return None
    
    def _create_observation(self, image: np.ndarray, robot_state: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """创建GR00T观察字典"""
        # 准备状态数组，添加时间维度
        left_arm = np.expand_dims(robot_state["left_arm"], axis=0)  # [1, 7]
        right_arm = np.expand_dims(robot_state["right_arm"], axis=0)  # [1, 7]
        left_ee = np.expand_dims(robot_state["left_ee"], axis=0)  # [1, 6]
        right_ee = np.expand_dims(robot_state["right_ee"], axis=0)  # [1, 6]
        
        obs = {
            "video.rs_view": image,  # [1, H, W, 3]
            "state.left_arm": left_arm,
            "state.right_arm": right_arm,
            "state.left_ee": left_ee,
            "state.right_ee": right_ee,
        }
        
        return obs
    
    def _action_to_full_joint_action(self, action_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """将GR00T动作字典转换为完整的关节动作张量
        
        Args:
            action_dict: GR00T输出的动作字典，包含：
                - action.left_arm: [16, 7]
                - action.right_arm: [16, 7]
                - action.left_ee: [16, 6]
                - action.right_ee: [16, 6]
        
        Returns:
            torch.Tensor: 完整的关节动作 [1, num_joints]
        """
        # 获取第一步动作（索引0）
        step_idx = 0  # 只使用第一步
        
        left_arm_action = action_dict["action.left_arm"][step_idx]  # [7]
        right_arm_action = action_dict["action.right_arm"][step_idx]  # [7]
        left_ee_action = action_dict["action.left_ee"][step_idx]  # [6]
        right_ee_action = action_dict["action.right_ee"][step_idx]  # [6]
        
        # 创建完整动作张量
        num_joints = len(self.all_joint_names)
        full_action = torch.zeros(num_joints, device=self.env.device, dtype=torch.float32)
        
        # 设置手臂动作
        if len(self.left_arm_indices) == 7:
            full_action[self.left_arm_indices] = torch.from_numpy(left_arm_action).to(self.env.device)
        if len(self.right_arm_indices) == 7:
            full_action[self.right_arm_indices] = torch.from_numpy(right_arm_action).to(self.env.device)
        
        # 设置Inspire手部动作
        if len(self.inspire_hand_indices) >= 12:
            # 组合左右手指动作
            ee_action = np.concatenate([right_ee_action, left_ee_action])  # [12]
            full_action[self.inspire_hand_indices[:12]] = torch.from_numpy(ee_action).to(self.env.device)
        
        # 返回批次维度 [1, num_joints]
        return full_action.unsqueeze(0)
    
    def get_action(self, env) -> Optional[torch.Tensor]:
        """获取动作（控制循环的核心方法）
        
        实现逻辑：
        1. 每步都获取当前图像和状态
        2. 如果当前动作序列已执行完，则进行新的推理
        3. 返回第一步动作
        
        Args:
            env: 环境实例
        
        Returns:
            torch.Tensor: 动作张量 [1, num_joints]，如果失败则返回None
        """
        try:
            # 如果当前动作序列为空或已执行完，进行新的推理
            if self.current_action_sequence is None:
                # 获取当前观察
                image = self._get_camera_image()
                if image is None:
                    print(f"[{self.name}] 无法获取图像，返回None")
                    return None
                
                robot_state = self._get_robot_state()
                observation = self._create_observation(image, robot_state)
                
                # GR00T推理
                print(f"[{self.name}] 进行GR00T推理...")
                self.current_action_sequence = self.policy.get_action(observation)
                self.action_step_index = 0
                print(f"[{self.name}] GR00T推理完成，动作序列长度: {len(next(iter(self.current_action_sequence.values()))))}")
            
            # 将动作转换为完整的关节动作
            full_action = self._action_to_full_joint_action(self.current_action_sequence)
            
            # 执行完第一步后，清空动作序列以触发下一次推理
            # 这样每次循环都会进行新的推理
            self.current_action_sequence = None
            
            return full_action
            
        except Exception as e:
            print(f"[{self.name}] 获取动作失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """清理资源"""
        print(f"[{self.name}] 清理GR00T VLA ActionProvider资源...")
        self.current_action_sequence = None
        # 注意：Gr00tPolicy可能不需要显式清理，但可以在这里添加必要的清理代码
        super().cleanup()

