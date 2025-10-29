#!/usr/bin/env python3
"""
VLA桥接客户端 - 连接unitree_sim_isaaclab和GR00T推理服务

此脚本在unitree_sim_isaaclab环境中运行，负责：
1. 从Isaac Sim获取图像和机器人状态
2. 发送到GR00T推理服务进行推理
3. 接收推理结果并转换为控制命令
4. 通过DDS发送控制命令给机器人

使用方法：
    # 在unitree_sim_isaaclab虚拟环境中运行
    python vla_bridge_client.py \
        --gr00t-host localhost \
        --gr00t-port 5556 \
        --gr00t-protocol zmq
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(project_root, 'unitree_sim_isaaclab'))

# 在导入Isaac Lab之前设置环境
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLA桥接客户端")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Cylinder-G129-Inspire-Joint",
                    help="Isaac Sim任务名称")
parser.add_argument("--gr00t-host", type=str, default="localhost",
                    help="GR00T推理服务主机地址")
parser.add_argument("--gr00t-port", type=int, default=5556,
                    help="GR00T推理服务端口号")
parser.add_argument("--gr00t-protocol", type=str, default="zmq",
                    choices=["zmq", "http"],
                    help="通信协议（zmq或http）")
parser.add_argument("--gr00t-api-token", type=str, default=None,
                    help="GR00T服务API令牌")
parser.add_argument("--control-freq", type=float, default=5.0,
                    help="控制频率（Hz）- VLA推理频率，建议5-10Hz以确保图像传输和动作执行完整")
parser.add_argument("--action-steps", type=int, default=6,
                    help="每个动作执行的物理步数，用于确保动作完全应用到机器人（default=6，对应约100ms）")
parser.add_argument("--enable-inspire-dds", action="store_true", default=True,
                    help="启用Inspire手部DDS")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 默认启用相机（VLA部署需要相机图像）
# AppLauncher会自动添加enable_cameras参数，如果没有传递，默认为False
# 我们确保它被设置为True
args_cli.enable_cameras = True

print(f"[配置] 相机已启用: {args_cli.enable_cameras}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 保存到全局，以便在清理时使用
import atexit
def cleanup_simulation_app():
    """清理仿真应用的函数"""
    try:
        if simulation_app is not None:
            simulation_app.close()
    except:
        pass

atexit.register(cleanup_simulation_app)

# 现在导入其他模块
import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import tasks

# 导入GR00T客户端（不使用gr00t模块，直接实现ZMQ客户端）
sys.path.insert(0, script_dir)
if args_cli.gr00t_protocol == "http":
    import requests
    import json_numpy
    json_numpy.patch()
else:
    # ZMQ协议需要msgpack
    import msgpack
    import io
    import zmq


class SimpleMsgSerializer:
    """简化的消息序列化器（不依赖gr00t模块）"""
    
    @staticmethod
    def encode_custom_classes(obj):
        """编码自定义类型为msgpack可序列化的格式"""
        # 处理torch.Tensor - 转换为numpy数组
        if hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
            # 是torch.Tensor
            try:
                obj = obj.detach().cpu().numpy()
            except:
                try:
                    obj = obj.cpu().numpy()
                except:
                    obj = np.array(obj)
        
        if isinstance(obj, np.ndarray):
            # 将numpy数组编码
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj
    
    @staticmethod
    def decode_custom_classes(obj):
        """解码自定义类型"""
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            # 解码numpy数组
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj
    
    @staticmethod
    def _convert_dict_recursively(obj):
        """递归转换字典中的所有torch.Tensor为numpy数组"""
        if isinstance(obj, dict):
            return {k: SimpleMsgSerializer._convert_dict_recursively(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [SimpleMsgSerializer._convert_dict_recursively(item) for item in obj]
        elif hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
            # torch.Tensor
            try:
                return obj.detach().cpu().numpy()
            except:
                try:
                    return obj.cpu().numpy()
                except:
                    return np.array(obj)
        else:
            return obj
    
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        """将字典序列化为bytes"""
        # 先递归转换所有torch.Tensor
        data = SimpleMsgSerializer._convert_dict_recursively(data)
        return msgpack.packb(data, default=SimpleMsgSerializer.encode_custom_classes)
    
    @staticmethod
    def from_bytes(data: bytes) -> dict:
        """将bytes反序列化为字典"""
        return msgpack.unpackb(data, object_hook=SimpleMsgSerializer.decode_custom_classes)


class GR00TClient:
    """GR00T推理服务客户端"""
    
    def __init__(self, host, port, protocol="zmq", api_token=None):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.api_token = api_token
        
        if protocol == "zmq":
            # 使用ZMQ客户端（不依赖gr00t模块）
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{host}:{port}")
            self.socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30秒超时
            print(f"[GR00T客户端] ZMQ客户端已连接到 {host}:{port}")
        else:
            # HTTP客户端不需要预先连接
            self.socket = None
            self.context = None
            print(f"[GR00T客户端] HTTP客户端将连接到 http://{host}:{port}")
    
    def _zmq_get_action(self, observation):
        """通过ZMQ获取动作（不依赖gr00t模块）"""
        # 构建请求
        request = {
            "endpoint": "get_action",
            "data": observation
        }
        if self.api_token:
            request["api_token"] = self.api_token
        
        # 发送请求
        try:
            self.socket.send(SimpleMsgSerializer.to_bytes(request))
            # 接收响应
            message = self.socket.recv()
            response = SimpleMsgSerializer.from_bytes(message)
            
            if "error" in response:
                raise RuntimeError(f"服务器错误: {response['error']}")
            
            return response
        except zmq.error.Again:
            raise TimeoutError(f"连接GR00T服务超时: {self.host}:{self.port}")
        except Exception as e:
            raise Exception(f"ZMQ通信失败: {e}")
    
    def get_action(self, observation):
        """发送观察并获取动作"""
        if self.protocol == "zmq":
            # 使用ZMQ直接通信（不依赖gr00t模块）
            return self._zmq_get_action(observation)
        else:
            # HTTP请求
            import requests
            import json_numpy
            json_numpy.patch()
            
            url = f"http://{self.host}:{self.port}/act"
            payload = {"observation": observation}
            
            if self.api_token:
                payload["api_token"] = self.api_token
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"HTTP请求失败: {response.status_code} - {response.text}")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'socket') and self.socket is not None:
            try:
                self.socket.close()
            except:
                pass
        if hasattr(self, 'context') and self.context is not None:
            try:
                self.context.term()
            except:
                pass


class VLABridge:
    """VLA桥接器 - 连接Isaac Sim和GR00T推理服务"""
    
    def __init__(self, env, gr00t_client):
        self.env = env
        self.gr00t_client = gr00t_client
        
        # 关节映射
        self._setup_joint_mapping()
        
        # 控制参数
        self.control_dt = 1.0 / args_cli.control_freq
        self.current_action_sequence = None
        self.action_step_index = 0
        self.step_count = 0  # 步骤计数器
        self.action_steps = args_cli.action_steps  # 每个动作执行的物理步数
        
        # 图像保存设置（用于debug）
        self.save_images = True
        self.max_saved_steps = 30  # 只保存前30步
        self.image_save_dir = Path("vla_debug_images") / datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.save_images:
            self.image_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[图像保存] 调试图像将保存到: {self.image_save_dir.absolute()}")
        
        print("[VLA桥接器] 初始化完成")
        print(f"  控制频率: {args_cli.control_freq} Hz")
        print(f"  每个动作执行{self.action_steps}个物理步（确保动作完全应用）")
    
    def _setup_joint_mapping(self):
        """设置关节映射"""
        self.all_joint_names = self.env.scene["robot"].data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}
        
        # 左臂和右臂关节
        left_arm_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        ]
        right_arm_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]
        
        self.left_arm_indices = [self.joint_to_index[name] for name in left_arm_names if name in self.joint_to_index]
        self.right_arm_indices = [self.joint_to_index[name] for name in right_arm_names if name in self.joint_to_index]
        
        # Inspire手部关节
        inspire_hand_names = [
            "R_pinky_proximal_joint", "R_ring_proximal_joint",
            "R_middle_proximal_joint", "R_index_proximal_joint",
            "R_thumb_proximal_pitch_joint", "R_thumb_proximal_yaw_joint",
            "L_pinky_proximal_joint", "L_ring_proximal_joint",
            "L_middle_proximal_joint", "L_index_proximal_joint",
            "L_thumb_proximal_pitch_joint", "L_thumb_proximal_yaw_joint",
        ]
        self.inspire_hand_indices = [self.joint_to_index[name] for name in inspire_hand_names if name in self.joint_to_index]
        
        print(f"[关节映射] 左臂: {len(self.left_arm_indices)}, 右臂: {len(self.right_arm_indices)}, 手部: {len(self.inspire_hand_indices)}")
    
    def _get_observation(self):
        """从环境获取观察"""
        # 获取图像
        if "front_camera" in self.env.scene.keys():
            image_tensor = self.env.scene["front_camera"].data.output["rgb"][0]
            image = image_tensor.cpu().numpy()
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # 保存图像用于debug（只保存前30步）
            if self.save_images and self.step_count < self.max_saved_steps:
                # image的形状是 [H, W, 3]，直接保存
                image_path = self.image_save_dir / f"step_{self.step_count:04d}.png"
                try:
                    # 使用PIL保存（如果没有PIL，尝试使用其他方法）
                    saved = False
                    try:
                        from PIL import Image
                        Image.fromarray(image).save(str(image_path))
                        saved = True
                    except ImportError:
                        # 如果没有PIL，使用opencv
                        try:
                            import cv2
                            # cv2使用BGR格式，需要转换
                            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(image_path), image_bgr)
                            saved = True
                        except ImportError:
                            # 如果都没有，使用numpy直接保存（imageio作为备选）
                            try:
                                import imageio
                                imageio.imwrite(str(image_path), image)
                                saved = True
                            except ImportError:
                                # 最后的备选：保存为numpy数组（可以用np.load加载）
                                np_path = str(image_path).replace('.png', '.npy')
                                np.save(np_path, image)
                                print(f"[图像保存] 使用numpy格式保存: {np_path}")
                                saved = True
                    if saved and image_path.exists():
                        print(f"[图像保存] Step {self.step_count}: {image_path.name} ({image.shape[0]}x{image.shape[1]})")
                except Exception as e:
                    print(f"[警告] 保存图像失败 (step {self.step_count}): {e}")
            
            image = np.expand_dims(image, axis=0)  # [1, H, W, 3]
        else:
            raise Exception("未找到front_camera")
        
        # 获取机器人状态
        joint_pos = self.env.scene["robot"].data.joint_pos[0]
        
        left_arm_state = joint_pos[self.left_arm_indices].cpu().numpy() if len(self.left_arm_indices) == 7 else np.zeros(7)
        right_arm_state = joint_pos[self.right_arm_indices].cpu().numpy() if len(self.right_arm_indices) == 7 else np.zeros(7)
        
        if len(self.inspire_hand_indices) >= 12:
            inspire_positions = joint_pos[self.inspire_hand_indices].cpu().numpy()
            right_ee_state = inspire_positions[:6] if len(inspire_positions) >= 6 else np.zeros(6)
            left_ee_state = inspire_positions[6:12] if len(inspire_positions) >= 12 else np.zeros(6)
        else:
            left_ee_state = np.zeros(6)
            right_ee_state = np.zeros(6)
        
        observation = {
            "video.rs_view": image,
            "state.left_arm": np.expand_dims(left_arm_state, axis=0),
            "state.right_arm": np.expand_dims(right_arm_state, axis=0),
            "state.left_ee": np.expand_dims(left_ee_state, axis=0),
            "state.right_ee": np.expand_dims(right_ee_state, axis=0),
        }
        
        # 确保所有值都是numpy数组（不是torch.Tensor）
        # 这对于序列化很重要
        for key, value in observation.items():
            if hasattr(value, 'cpu') and hasattr(value, 'numpy'):
                # 如果是torch.Tensor，转换为numpy
                observation[key] = value.detach().cpu().numpy()
            elif not isinstance(value, np.ndarray):
                # 如果不是numpy数组，尝试转换
                try:
                    observation[key] = np.array(value)
                except Exception as e:
                    print(f"[警告] 无法转换 {key} 为numpy数组: {e}")
        
        return observation
    
    def _execute_action(self, action_dict):
        """执行动作（第一步）"""
        step_idx = 0
        
        left_arm_action = action_dict["action.left_arm"][step_idx]  # [7]
        right_arm_action = action_dict["action.right_arm"][step_idx]  # [7]
        left_ee_action = action_dict["action.left_ee"][step_idx]  # [6]
        right_ee_action = action_dict["action.right_ee"][step_idx]  # [6]
        
        # 创建完整的关节动作张量
        num_joints = len(self.all_joint_names)
        full_action = torch.zeros(num_joints, device=self.env.device, dtype=torch.float32)
        
        # 设置手臂动作
        if len(self.left_arm_indices) == 7:
            full_action[self.left_arm_indices] = torch.from_numpy(left_arm_action).to(self.env.device)
        if len(self.right_arm_indices) == 7:
            full_action[self.right_arm_indices] = torch.from_numpy(right_arm_action).to(self.env.device)
        
        # 设置Inspire手部动作
        if len(self.inspire_hand_indices) >= 12:
            # 组合左右手指动作：右手指6个 + 左手指6个 = 12个
            ee_action = np.concatenate([right_ee_action, left_ee_action])  # [12]
            full_action[self.inspire_hand_indices[:12]] = torch.from_numpy(ee_action).to(self.env.device)
        
        # 直接通过环境接口设置关节目标
        # 使用unsqueeze添加batch维度
        full_action = full_action.unsqueeze(0)  # [1, num_joints]
        
        # 返回完整的动作张量 [1, num_joints]
        return full_action
    
    def step(self):
        """执行一步控制"""
        try:
            # === 第一步：获取当前观察（确保图像是最新的） ===
            print(f"\n[控制循环] 步骤 {self.step_count}")
            observation_start = time.time()
            observation = self._get_observation()
            observation_time = time.time() - observation_start
            print(f"[观察] 获取观察耗时: {observation_time*1000:.2f}ms")
            
            # === 第二步：GR00T推理 ===
            print("[推理] 发送观察到GR00T服务...")
            inference_start = time.time()
            action_dict = self.gr00t_client.get_action(observation)
            inference_time = time.time() - inference_start
            print(f"[推理] 推理完成，耗时: {inference_time*1000:.2f}ms")
            
            # 打印动作序列信息
            print("[动作序列]")
            for key, value in action_dict.items():
                if isinstance(value, np.ndarray):
                    shape_str = f"形状: {value.shape}"
                    # 显示第一步的值
                    if len(value.shape) >= 1:
                        first_step = value[0] if value.shape[0] > 0 else value
                        if isinstance(first_step, np.ndarray):
                            values_str = f"第一步: {first_step}"
                            # 如果值太多，只显示前几个
                            if len(first_step) > 10:
                                values_str = f"第一步 (前10个): {first_step[:10]} ... (共{len(first_step)}个)"
                        else:
                            values_str = f"第一步: {first_step}"
                    else:
                        values_str = f"值: {value}"
                    print(f"  {key}: {shape_str}, {values_str}")
                else:
                    print(f"  {key}: {value}")
            
            # === 第三步：转换动作 ===
            full_action = self._execute_action(action_dict)
            
            # 打印执行的完整动作（只显示非零或手臂/手部关节）
            print("[执行动作]")
            num_nonzero = torch.nonzero(full_action[0]).numel()
            print(f"  完整动作形状: {full_action.shape}")
            print(f"  非零动作数: {num_nonzero}/{full_action.numel()}")
            
            # 显示手臂和手部动作值
            if len(self.left_arm_indices) == 7:
                left_arm_action = full_action[0, self.left_arm_indices].cpu().numpy()
                print(f"  左臂动作: {left_arm_action}")
            if len(self.right_arm_indices) == 7:
                right_arm_action = full_action[0, self.right_arm_indices].cpu().numpy()
                print(f"  右臂动作: {right_arm_action}")
            if len(self.inspire_hand_indices) >= 12:
                hand_action = full_action[0, self.inspire_hand_indices[:12]].cpu().numpy()
                print(f"  手部动作 (前6个): {hand_action[:6]} ... (共{len(hand_action)}个)")
            
            # === 第四步：执行动作（执行多个物理步以确保动作完全应用） ===
            print(f"[执行] 开始执行动作（{self.action_steps}个物理步）...")
            execution_start = time.time()
            for i in range(self.action_steps):
                # 每个物理步都使用相同的动作（位置控制会平滑过渡到目标位置）
                obs, rew, terminated, truncated, info = self.env.step(full_action)
                
                if terminated or truncated:
                    print(f"[执行] 环境已终止/截断，停止执行")
                    break
            
            execution_time = time.time() - execution_start
            avg_step_time = execution_time / self.action_steps if self.action_steps > 0 else 0
            print(f"[执行] 动作执行完成，总耗时: {execution_time*1000:.2f}ms (平均每步: {avg_step_time*1000:.2f}ms)")
            
            # === 等待图像更新（给渲染器一些时间更新图像） ===
            # Isaac Sim的渲染可能在step之后异步更新，所以稍微等待一下
            time.sleep(0.01)  # 10ms等待，确保图像更新
            
            self.step_count += 1
            
            return True
            
        except Exception as e:
            print(f"[控制循环] 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """运行控制循环"""
        print("\n" + "=" * 60)
        print("VLA控制循环开始")
        print(f"控制频率: {args_cli.control_freq} Hz")
        print("按 Ctrl+C 停止")
        print("=" * 60 + "\n")
        
        while True:
            start_time = time.time()
            
            if not self.step():
                break
            
            # 控制频率
            elapsed = time.time() - start_time
            sleep_time = max(0, self.control_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    print("=" * 60)
    print("VLA桥接客户端")
    print("=" * 60)
    print(f"任务: {args_cli.task}")
    print(f"GR00T服务: {args_cli.gr00t_protocol}://{args_cli.gr00t_host}:{args_cli.gr00t_port}")
    print("=" * 60)
    
    # 创建环境
    print("\n创建Isaac Sim环境...")
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        print("环境创建成功")
    except Exception as e:
        print(f"创建环境失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 可选：初始化DDS对象（消除警告，但不一定需要用于控制）
    if args_cli.enable_inspire_dds or args_cli.robot_type == "g129":
        print("\n初始化DDS对象...")
        try:
            from dds.dds_create import create_dds_objects
            # 设置必要的参数
            args_cli.enable_dex1_dds = False
            args_cli.enable_dex3_dds = False
            args_cli.enable_wholebody_dds = False
            reset_pose_dds, sim_state_dds, dds_manager = create_dds_objects(args_cli, env)
            print("DDS对象初始化成功（用于状态发布）")
        except Exception as e:
            print(f"DDS初始化警告（可忽略，不影响核心功能）: {e}")
    
    # 创建GR00T客户端
    print(f"\n连接到GR00T推理服务...")
    try:
        gr00t_client = GR00TClient(
            host=args_cli.gr00t_host,
            port=args_cli.gr00t_port,
            protocol=args_cli.gr00t_protocol,
            api_token=args_cli.gr00t_api_token
        )
    except Exception as e:
        print(f"连接GR00T服务失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建VLA桥接器
    print("\n创建VLA桥接器...")
    bridge = VLABridge(env, gr00t_client)
    
    # 重置环境
    env.reset()
    
    # 运行控制循环
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止...")
    except Exception as e:
        print(f"\n运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n清理资源...")
        try:
            # 先停止环境
            if 'env' in locals() and env is not None:
                try:
                    env.close()
                except Exception as e:
                    print(f"关闭环境时出现警告（可忽略）: {e}")
        except Exception as e:
            print(f"清理环境时出现警告（可忽略）: {e}")
        
        # 关闭仿真应用（需要全局访问simulation_app）
        try:
            if 'simulation_app' in globals() and simulation_app is not None:
                simulation_app.close()
        except Exception as e:
            print(f"关闭仿真应用时出现警告（可忽略）: {e}")
        
        print("清理完成（部分警告可以忽略）")


if __name__ == "__main__":
    main()

