#!/usr/bin/env python3
"""
VLA部署主脚本 - 将Isaac-GR00T模型部署到Isaac Sim仿真环境

使用方式:
    python run_vla_deployment.py \
        --checkpoint checkpoint-59900 \
        --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
        --device cuda

或者使用默认配置:
    python run_vla_deployment.py
"""

import os
import sys
import argparse

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROJECT_ROOT"] = project_root

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))  # Isaac-GR00T目录
sys.path.insert(0, os.path.join(project_root, 'unitree_sim_isaaclab'))

# 在导入Isaac Lab相关模块之前设置环境
from isaaclab.app import AppLauncher

# 命令行参数
parser = argparse.ArgumentParser(description="VLA部署脚本 - Isaac-GR00T模型部署")
parser.add_argument("--checkpoint", type=str, default="checkpoint-59900",
                    help="GR00T模型检查点路径（相对于Isaac-GR00T目录）")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Cylinder-G129-Inspire-Joint",
                    help="Isaac Sim任务名称")
parser.add_argument("--device", type=str, default=None,
                    choices=["cuda", "cpu"],
                    help="计算设备（默认自动检测）")
parser.add_argument("--denoising_steps", type=int, default=4,
                    help="GR00T去噪步数")
parser.add_argument("--step_hz", type=int, default=30,
                    help="控制频率（Hz），建议30Hz以便给模型推理留出时间")
parser.add_argument("--enable_cameras", action="store_true", default=True,
                    help="启用相机")
parser.add_argument("--enable_inspire_dds", action="store_true", default=True,
                    help="启用Inspire手部DDS")
parser.add_argument("--robot_type", type=str, default="g129",
                    help="机器人类型")

# 添加AppLauncher参数
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# 初始化AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 现在导入其他模块
import gymnasium as gym
import torch
import time
import signal

from layeredcontrol.robot_control_system import RobotController, ControlConfig
from action_provider.action_base import ActionProvider
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import tasks

# 导入GR00T VLA ActionProvider
sys.path.insert(0, script_dir)
from vla_deployment_gr00t_action_provider import GR00TVLAActionProvider

# 调整checkpoint路径
if not os.path.isabs(args_cli.checkpoint):
    checkpoint_path = os.path.join(os.path.dirname(script_dir), args_cli.checkpoint)
else:
    checkpoint_path = args_cli.checkpoint

if not os.path.exists(checkpoint_path):
    print(f"错误: 检查点路径不存在: {checkpoint_path}")
    sys.exit(1)

print("=" * 60)
print("Isaac-GR00T VLA 部署脚本")
print("=" * 60)
print(f"检查点路径: {checkpoint_path}")
print(f"任务: {args_cli.task}")
print(f"控制频率: {args_cli.step_hz} Hz")
print("=" * 60)


def setup_signal_handlers(controller):
    """设置信号处理器"""
    def signal_handler(signum, frame):
        print(f"\n收到信号 {signum}，正在停止控制器...")
        try:
            controller.stop()
        except Exception as e:
            print(f"停止控制器失败: {e}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """主函数"""
    print("\n解析环境配置...")
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
    except Exception as e:
        print(f"解析环境配置失败: {e}")
        return
    
    print("\n创建环境...")
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        print("环境创建成功")
    except Exception as e:
        print(f"创建环境失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建控制配置
    print("\n创建控制配置...")
    control_config = ControlConfig()
    control_config.step_hz = args_cli.step_hz
    control_config.use_rl_action_mode = False  # 使用位置控制模式
    
    # 创建GR00T VLA ActionProvider
    print("\n创建GR00T VLA ActionProvider...")
    try:
        action_provider = GR00TVLAActionProvider(
            env=env,
            checkpoint_path=checkpoint_path,
            device=args_cli.device,
            denoising_steps=args_cli.denoising_steps
        )
        print("GR00T VLA ActionProvider创建成功")
    except Exception as e:
        print(f"创建GR00T VLA ActionProvider失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建控制器
    print("\n创建控制器...")
    try:
        controller = RobotController(env, control_config)
        controller.set_action_provider(action_provider)
        print("控制器创建成功")
    except Exception as e:
        print(f"创建控制器失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 设置信号处理器
    setup_signal_handlers(controller)
    
    try:
        # 启动控制器
        print("\n启动控制器...")
        print("=" * 60)
        print("控制循环开始")
        print("按 Ctrl+C 停止")
        print("=" * 60)
        
        controller.start()
        
        # 运行控制循环
        controller.run()
        
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止...")
    except Exception as e:
        print(f"\n运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n清理资源...")
        try:
            controller.stop()
            action_provider.cleanup()
        except Exception as e:
            print(f"清理失败: {e}")
        print("程序退出")


if __name__ == "__main__":
    main()

