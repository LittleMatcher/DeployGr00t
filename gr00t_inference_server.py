#!/usr/bin/env python3
"""
GR00T推理服务 - 独立运行的推理服务器

此服务可以独立运行在Isaac-GR00T虚拟环境中，通过ZMQ或HTTP接收推理请求并返回结果。

使用方法：
    # ZMQ模式（默认）
    python gr00t_inference_server.py \
        --checkpoint checkpoint-59900 \
        --port 5556 \
        --device cuda

    # HTTP模式
    python gr00t_inference_server.py \
        --checkpoint checkpoint-59900 \
        --port 8000 \
        --http-server \
        --device cuda
"""

import sys
import os
import argparse

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.video import VideoToTensor, VideoResize, VideoToNumpy
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.transforms import GR00TTransform
import torch


def convert_tensors_to_numpy(obj):
    """递归将torch.Tensor转换为numpy数组"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: convert_tensors_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_numpy(item) for item in obj]
    else:
        return obj


class CustomDataConfig:
    """自定义数据配置，匹配 checkpoint-59900 的 metadata"""
    
    def __init__(self):
        self.video_keys = ["video.rs_view"]
        self.state_keys = ["state.left_arm", "state.right_arm", "state.left_ee", "state.right_ee"]
        self.action_keys = ["action.left_arm", "action.right_arm", "action.left_ee", "action.right_ee"]
        self.language_keys = []
        self.observation_indices = [0]
        self.action_indices = list(range(16))
    
    def modality_config(self):
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
    
    def transform(self):
        """创建变换管道"""
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=224, width=224),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys}
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


def main():
    parser = argparse.ArgumentParser(description="GR00T推理服务")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-59900",
                        help="检查点路径（相对于Isaac-GR00T目录）")
    parser.add_argument("--port", type=int, default=5556,
                        help="服务端口号")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="服务主机地址")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"],
                        help="计算设备（默认自动检测）")
    parser.add_argument("--denoising_steps", type=int, default=4,
                        help="去噪步数")
    parser.add_argument("--http-server", action="store_true",
                        help="使用HTTP服务器模式（默认ZMQ）")
    parser.add_argument("--api-token", type=str, default=None,
                        help="API令牌（可选）")
    
    args = parser.parse_args()
    
    # 调整checkpoint路径
    if not os.path.isabs(args.checkpoint):
        checkpoint_path = os.path.join(os.path.dirname(script_dir), args.checkpoint)
    else:
        checkpoint_path = args.checkpoint
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点路径不存在: {checkpoint_path}")
        sys.exit(1)
    
    # 设置设备
    if args.device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("GR00T推理服务")
    print("=" * 60)
    print(f"检查点路径: {checkpoint_path}")
    print(f"服务地址: {args.host}:{args.port}")
    print(f"计算设备: {device}")
    print(f"服务模式: {'HTTP' if args.http_server else 'ZMQ'}")
    print("=" * 60)
    
    # 创建数据配置
    data_config = CustomDataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    # 加载模型
    print("\n正在加载GR00T模型...")
    try:
        policy = Gr00tPolicy(
            model_path=checkpoint_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag="new_embodiment",
            denoising_steps=args.denoising_steps,
            device=device,
        )
        print("模型加载完成!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 创建包装函数，将torch.Tensor转换为numpy数组
    def get_action_wrapper(observation):
        """包装policy.get_action，确保返回numpy数组"""
        action = policy.get_action(observation)
        return convert_tensors_to_numpy(action)
    
    # 启动服务器
    print(f"\n启动推理服务...")
    if args.http_server:
        from gr00t.eval.http_server import HTTPInferenceServer
        server = HTTPInferenceServer(
            policy=policy,
            port=args.port,
            host=args.host,
            api_token=args.api_token
        )
        print("HTTP服务器已启动，等待推理请求...")
        server.run()
    else:
        from gr00t.eval.service import BaseInferenceServer
        
        # 创建自定义服务器，使用包装的get_action
        class CustomRobotInferenceServer(BaseInferenceServer):
            """自定义服务器，自动转换torch.Tensor为numpy数组"""
            def __init__(self, policy, host="*", port=5555, api_token=None):
                super().__init__(host, port, api_token)
                # 注册包装后的get_action
                self.register_endpoint("get_action", get_action_wrapper)
                # 注册modality_config端点
                self.register_endpoint(
                    "get_modality_config", policy.get_modality_config, requires_input=False
                )
        
        server = CustomRobotInferenceServer(
            policy,
            host=args.host,
            port=args.port,
            api_token=args.api_token
        )
        print("ZMQ服务器已启动，等待推理请求...")
        server.run()


if __name__ == "__main__":
    main()

