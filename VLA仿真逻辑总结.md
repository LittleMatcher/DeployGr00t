# VLA仿真部署系统逻辑总结

## 一、系统架构概述

这是一个基于**分布式架构**的Vision-Language-Action (VLA) 机器人控制仿真系统，将三个独立的组件通过通信协议连接起来：

```
┌─────────────────────────────────────────────────────────┐
│            单元1: unitree_sim_isaaclab                  │
│  ┌────────────────────────────────────────────────┐    │
│  │  Isaac Sim仿真环境                              │    │
│  │  - 运行Isaac-PickPlace-Cylinder-G129任务       │    │
│  │  - Unitree G1机器人模型                         │    │
│  │  - 提供front_camera图像                         │    │
│  │  - 提供机器人关节状态                           │    │
│  └────────────────────────────────────────────────┘    │
│                         ↕                               │
│  ┌────────────────────────────────────────────────┐    │
│  │  vla_bridge_client.py (桥接客户端)              │    │
│  │  - 获取图像和状态                              │    │
│  │  - 调用GR00T推理                               │    │
│  │  - 执行控制动作                                │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                         ↕ ZMQ/HTTP
┌─────────────────────────────────────────────────────────┐
│            单元2: Isaac-GR00T                           │
│  ┌────────────────────────────────────────────────┐    │
│  │  gr00t_inference_server.py (推理服务)           │    │
│  │  - 加载checkpoint-59900模型                    │    │
│  │  - 接收观察数据                                │    │
│  │  - 执行VLA推理                                 │    │
│  │  - 返回动作序列                                │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 二、核心组件职责

### 2.1 GR00T推理服务 (`gr00t_inference_server.py`)

**运行环境**: Isaac-GR00T虚拟环境

**职责**:
1. **模型加载**: 加载微调后的checkpoint-59900模型
2. **推理接口**: 提供ZMQ或HTTP接口接收推理请求
3. **处理观察**: 接收图像和机器人状态数据
4. **生成动作**: 使用GR00T模型预测动作序列（通常8步）
5. **序列化输出**: 将torch.Tensor转换为numpy数组并序列化返回

**关键代码逻辑**:
- 加载Gr00tPolicy模型
- 监听端口（默认5556 for ZMQ, 8000 for HTTP）
- 接收observation字典，包含:
  - `video.rs_view`: 图像 [1, H, W, 3]
  - `state.left_arm`: 左臂状态 [1, 7]
  - `state.right_arm`: 右臂状态 [1, 7]
  - `state.left_ee`: 左手状态 [1, 6]
  - `state.right_ee`: 右手状态 [1, 6]
- 调用`policy.get_action(observation)`生成动作序列
- 返回动作字典，包含:
  - `action.left_arm`: [8, 7] - 8步左臂动作序列
  - `action.right_arm`: [8, 7] - 8步右臂动作序列
  - `action.left_ee`: [8, 6] - 8步左手动作序列
  - `action.right_ee`: [8, 6] - 8步右手动作序列

### 2.2 VLA桥接客户端 (`vla_bridge_client.py`)

**运行环境**: unitree_sim_isaaclab虚拟环境

**职责**:
1. **初始化Isaac Sim环境**: 创建Isaac-PickPlace-Cylinder-G129-Inspire-Joint任务
2. **获取观察数据**: 从仿真环境获取图像和机器人状态
3. **与GR00T通信**: 发送观察数据，接收动作序列
4. **动作转换**: 将GR00T输出的动作序列转换为Isaac Sim的关节控制命令
5. **执行控制**: 将动作应用到仿真环境

**关键代码逻辑**:

#### 2.2.1 观察获取 (`_get_observation`)
```python
# 1. 获取相机图像
image = env.scene["front_camera"].data.output["rgb"][0]  # [H, W, 3]
# 转换为uint8格式，添加时间维度 -> [1, H, W, 3]

# 2. 获取机器人关节状态
joint_pos = env.scene["robot"].data.joint_pos[0]

# 3. 提取关键状态
left_arm_state = joint_pos[left_arm_indices]  # 7个关节
right_arm_state = joint_pos[right_arm_indices]  # 7个关节
left_ee_state = inspire_hand_indices[6:12]  # 左手6个关节
right_ee_state = inspire_hand_indices[0:6]  # 右手6个关节

# 4. 构建observation字典
observation = {
    "video.rs_view": image,  # [1, H, W, 3]
    "state.left_arm": left_arm_state,  # [1, 7]
    "state.right_arm": right_arm_state,  # [1, 7]
    "state.left_ee": left_ee_state,  # [1, 6]
    "state.right_ee": right_ee_state,  # [1, 古老]
}
```

#### 2.2.2 动作执行 (`_execute_action`)
```python
# 1. 提取动作序列的第一步
left_arm_action = action_dict["action.left_arm"][0]  # [7]
right_arm_action = action_dict["action.right_arm"][0]  # [7]
left_ee_action = action_dict["action.left_ee"][0]  # [6]
right_ee_action = action_dict["action.right_ee"][0]  # [6]

# 2. 映射到完整关节空间（29个关节）
full_action = torch.zeros(29)  # 零初始化
full_action[left_arm_indices] = left_arm_action  # 7个
full_action[right_arm_indices] = right_arm_action  # 7个
full_action[inspire_hand_indices] = [right_ee_action, left_ee_action]  # 12个

# 3. 返回 [1, 29] 的动作张量
return full_action.unsqueeze(0)
```

### 2.3 控制循环 (`step`方法)

**核心流程**:

```
┌─────────────────────────────────────────────────────┐
│  步骤1: 获取当前观察                                │
│  - 从Isaac Sim获取图像（front_camera）              │
│  - 从机器人获取关节状态                            │
│  - 保存图像用于debug（前30步）                     │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│  步骤2: GR00T推理                                   │
│  - 构建observation字典                              │
│  - 通过ZMQ/HTTP发送到GR00T服务                     │
│  - 接收动作序列（8步）                             │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│  步骤3: 动作转换                                    │
│  - 提取动作序列的第一步                             │
│  - 映射到29个关节空间                              │
│  - 构建完整动作张量                                │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│  步骤4: 执行动作（多次物理步）                      │
│  - 执行6个物理步（默认）确保动作完全应用            │
│  - 每步都使用相同的目标位置（位置控制会平滑过渡）   │
│  - 检查是否终止/截断                               │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│  步骤5: 等待渲染更新                                │
│  - 等待10ms确保图像渲染完成                        │
│  - 准备下一次循环                                  │
└─────────────────────────────────────────────────────┘
```

## 三、数据流详解

### 3.1 观察数据流 (Environment → GR00T)

```
Isaac Sim环境
    ↓
[获取图像] front_camera.rgb → numpy [H, W, 3] uint8
    ↓ 添加时间维度
[1, H, W, 3]
    ↓
[获取状态] robot.joint_pos → torch.Tensor
    ↓ 提取关键关节
[左臂7 + 右臂7 + 左手6 + 右手6]
    ↓ 转换为numpy
observation字典
    ↓ 序列化 (msgpack/JSON)
ZMQ/HTTP传输
    ↓
GR00T推理服务
```

### 3.2 动作数据流 (GR00T → Environment)

```
GR00T模型推理
    ↓
动作序列 [8步] × [左臂7 + 右臂7 + 左手6 + 右手6]
    ↓ 序列化 (msgpack/JSON)
ZMQ/HTTP传输
    ↓ 反序列化
动作字典
    ↓ 提取第一步
单步动作 [左臂7 + 右臂7 + 左手6 + 右手6]
    ↓ 映射到29关节空间
full_action [1, 29]
    ↓ 执行6个物理步
Isaac Sim环境更新
```

## 四、关键技术点

### 4.1 分布式通信

- **ZMQ协议**: 使用REQ-REP模式，msgpack序列化，低延迟高效
- **HTTP协议**: REST API，JSON序列化，易于调试
- **序列化**: 关键是将torch.Tensor转换为numpy.ndarray，支持嵌套字典/列表

### 4.2 数据格式转换

**问题**: torch.Tensor不能直接序列化

**解决方案**:
1. 在发送前：`tensor.detach().cpu().numpy()`
2. 在序列化前：递归转换字典/列表中的所有Tensor
3. 使用自定义序列化器：`SimpleMsgSerializer` / `MsgSerializer`

### 4.3 控制频率优化

- **控制频率**: 5 Hz（默认），降低到确保图像传输和动作执行完整
- **动作步数**: 每个动作执行6个物理步（约100ms），确保动作完全应用到机器人
- **同步机制**: 执行完动作后等待10ms，确保图像渲染完成

### 4.4 关节映射

**G1机器人关节结构**:
- 左臂: 7个关节 (肩部3 + 肘部1 + 腕部3)
- 右臂: 7个关节
- Inspire手部: 12个关节 (每手6个: 右手0-5, 左手6-11)
- 总计: 29个关节（G1本体20 + Inspire手部9，实际上可能有更多）

**映射逻辑**:
```python
# 从完整关节列表中找到对应索引
left_arm_indices = [joint_to_index[name] for name in left_arm_names]
right_arm_indices = [joint_to_index[name] for name in right_arm_names]
inspire_hand_indices = [joint_to_index[name] for name in inspire_hand_names]

# 构建动作时填充对应位置
full_action[left_arm_indices] = left_arm_action
full_action[right_arm_indices] = right_arm_action
full_action[inspire_hand_indices] = ee_action
```

## 五、执行时序

### 5.1 单步执行时间线

```
T0: 获取观察 (~10ms)
    - 读取相机图像
    - 读取关节状态
    - 保存debug图像（如果需要）

T1: GR00T推理 (~50ms)
    - 序列化observation
    - 网络传输（ZMQ/HTTP）
    - 模型推理
    - 序列化action
    - 网络传输返回

T2: 动作转换 (~1ms)
    - 提取第一步动作
    - 映射到关节空间

T3: 执行动作 (~100ms)
    - 6个物理步 × ~16.67ms/步

T4: 等待渲染 (~10ms)
    - 确保图像更新

总时间: ~171ms (对应约5.8 Hz)
```

### 5.2 控制循环频率

```
控制频率: 5 Hz (200ms周期)
├─ 实际执行时间: ~171ms
└─ 额外等待: ~29ms (保持频率稳定)
```

## 六、调试功能

### 6.1 图像保存

- **位置**: `vla_debug_images/YYYYMMDD_HHMMSS/`
- **格式**: PNG格式，命名`step_0000.png`到`step_0029.png`
- **用途**: 验证图像传输，调试推理问题

### 6.2 日志输出

每步输出：
- 观察获取耗时
- 推理时间和动作序列信息
- 执行动作的详细信息（形状、非零值、关键关节值）
- 动作执行耗时

## 七、部署流程

### 7.1 启动顺序

```
1. 启动GR00T推理服务 (Isaac-GR00T环境)
   ↓
2. 等待服务就绪（显示"Server is ready"）
   ↓
3. 启动VLA桥接客户端 (unitree_sim_isaaclab环境)
   ↓
4. 建立连接，开始控制循环
```

### 7.2 启动命令

```bash
# 终端1: GR00T推理服务
cd Isaac-GR00T
python deployment_scripts/gr00t_inference_server.py \
    --checkpoint checkpoint-59900 \
    --port 5556 \
    --device cuda

# 终端2: VLA桥接客户端
cd unitree_sim_isaaclab
python ../Isaac-GR00T/deployment_scripts/vla_bridge_client.py \
    --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
    --gr00t-host localhost \
    --gr00t-port 5556 \
    --gr00t-protocol zmq \
    --control-freq 5.0 \
    --action-steps 6
```

## 八、关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--control-freq` | 5.0 Hz | VLA推理频率，建议5-10Hz |
| `--action-steps` | 6 | 每个动作执行的物理步数 |
| `--gr00t-port` | 5556 | GR00T服务端口 |
| `--gr00t-protocol` | zmq | 通信协议（zmq/http） |

## 九、总结

这是一个**闭环控制系统**，实现了：

1. **视觉感知**: 从Isaac Sim获取机器人相机图像
2. **状态感知**: 获取机器人关节状态
3. **智能决策**: 使用微调的GR00T模型生成动作序列
4. **精确执行**: 将动作转换为关节控制命令并执行
5. **循环迭代**: 执行后获取新的观察，继续下一次决策

系统通过**分布式架构**解耦了仿真环境和模型推理，使得：
- 各组件可以在不同虚拟环境中运行
- 模型推理可以独立扩展和优化
- 通信协议保证了低延迟和高可靠性

整个系统实现了"**感知-决策-执行**"的完整闭环，使机器人能够基于视觉和状态信息自主完成任务。

