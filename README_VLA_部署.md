# Isaac-GR00T VLA模型部署指南

本指南介绍如何将微调后的Isaac-GR00T模型部署到Isaac Sim仿真环境中。

## 概述

部署流程实现了以下控制循环：

1. **获取观察**：从Isaac Sim仿真环境获取Unitree G1机器人的头部相机图像和关节状态
2. **模型推理**：将观察输入到Isaac-GR00T模型，获取16步的动作序列
3. **执行动作**：从动作序列中选取第一步，发送给机器人控制器执行
4. **循环往复**：执行完第一步后，返回步骤1，使用新的观察进行下一次推理

## 文件说明

### 核心文件

- `vla_deployment_gr00t_action_provider.py`: GR00T VLA ActionProvider实现
  - 继承自`ActionProvider`基类
  - 实现了从图像和状态到机器人控制动作的完整流程
  - 自动处理图像预处理、状态提取、模型推理和动作转换

- `run_vla_deployment.py`: 主部署脚本
  - 创建Isaac Sim环境
  - 初始化GR00T模型和ActionProvider
  - 启动控制循环

## 使用方法

### 基本使用

```bash
cd Isaac-GR00T
python deployment_scripts/run_vla_deployment.py \
    --checkpoint checkpoint-59900 \
    --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
    --device cuda
```

### 参数说明

- `--checkpoint`: GR00T模型检查点路径（相对于Isaac-GR00T目录）
- `--task`: Isaac Sim任务名称，应与数据采集时使用的任务一致
- `--device`: 计算设备，`cuda`或`cpu`（默认自动检测）
- `--denoising_steps`: GR00T去噪步数（默认4）
- `--step_hz`: 控制频率，建议30Hz（默认30）

### 完整示例

```bash
# 使用CUDA加速
python deployment_scripts/run_vla_deployment.py \
    --checkpoint checkpoint-59900 \
    --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
    --device cuda \
    --denoising_steps 4 \
    --step_hz 30

# 使用CPU（较慢）
python deployment_scripts/run_vla_deployment.py \
    --checkpoint checkpoint-59900 \
    --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
    --device cpu \
    --step_hz 10
```

## 部署流程详解

### 1. 观察获取

ActionProvider从Isaac Sim环境获取：

- **图像观察**：`front_camera`一致RGB图像（头部相机视角）
- **状态观察**：
  - 左臂7个关节角度 (`left_arm`)
  - 右臂7个关节角度 (`right_arm`)
  - 左手指6个关节状态 (`left_ee`) - 从12个Inspire关节中取前6个
  - 右手指6个关节状态 (`right_ee`) - 从12个Inspire关节中取后6个

### 2. 模型推理

使用加载的GR00T模型进行推理：

```python
observation = {
    "video.rs_view": image,  # [1, H, W, 3]
    "state.left_arm": left_arm_state,  # [1, 7]
    "state.right_arm": right_arm_state,  # [1, 7]
    "state.left_ee": left_ee_state,  # [1, 6]
    "state.right_ee": right_ee_state,  # [1, 6]
}

action_sequence = policy.get_action(observation)
# 返回:
# {
#     "action.left_arm": [16, 7],
#     "action.right_arm": [16, 7],
#     "action.left_ee": [16, 6],
#     "action.right_ee": [16, 6],
# }
```

### 3. 动作执行

从16步动作序列中选取第一步（索引0），转换为完整的关节动作：

```python
# 提取第一步动作
left_arm_action = action_sequence["action.left_arm"][0]  # [7]
right_arm_action = action_sequence["action.right_arm"][0]  # [7]
left_ee_action = action_sequence["action.left_ee"][0]  # [6]
right_ee_action = action_sequence["action.right_ee"][0]  # [6]

# 转换为完整关节动作张量
full_action = torch.zeros(num_joints)
full_action[left_arm_indices] = left_arm_action
full_action[right_arm_indices] = right_arm_action
full_action[inspire_hand_indices] = [right_ee_action, left_ee_action]
```

### 4. 控制循环

每次`get_action()`调用都会：
1. 获取当前图像和状态
2. 执行GR00T推理（如果当前动作序列已执行完）
3. 返回第一步动作
4. 清空当前动作序列，以便下次循环时进行新的推理

## 注意事项

### 性能优化

1. **控制频率**：建议使用30Hz或更低，给模型推理留出足够时间
2. **GPU加速**：使用CUDA可以显著提升推理速度
3. **去噪步数**：减少`denoising_steps`可以加快推理，但可能降低动作质量

### 调试建议

1. **检查模型加载**：确保checkpoint路径正确，模型文件完整
2. **检查环境配置**：确保任务名称与数据采集时一致
3. **检查关节映射**：确保机器人关节名称匹配（特别是Inspire手部关节）
4. **监控推理时间**：如果推理时间过长，考虑降低控制频率或优化模型

### 常见问题

**Q: 图像获取失败**
- 检查环境中是否有`front_camera`
- 确保`--enable_cameras`参数已启用

**Q: 关节索引错误**
- 检查机器人配置是否正确
- 确认使用的关节名称与实际机器人模型匹配

**Q: 动作执行异常**
- 检查动作值的范围是否合理
- 确认动作空间是否与训练时一致

## 与xr_teleoperate集成

虽然当前实现直接在Isaac Sim中获取图像和控制机器人，但如果需要与`xr_teleoperate`集成，可以：

1. 使用`xr_teleoperate/teleop/image_server`的`ImageClient`从Isaac Sim获取图像
2. 使用`xr_teleoperate/teleop/robot_control`的`G1_29_ArmController`通过DDS控制机器人

这需要在`GR00TVLAActionProvider`中修改图像获取和动作执行的部分。

## 下一步

- 添加动作平滑和插值
- 支持动作序列的重放（使用更多步而不只是第一步）
- 添加性能监控和日志记录
- 支持多环境并行部署

