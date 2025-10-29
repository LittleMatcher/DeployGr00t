# 分布式VLA部署指南

本指南介绍如何将三个程序（xr_teleoperate、unitree_sim_isaaclab、Isaac-GR00T）通过通信协议连接起来进行分布式部署。

## 架构概览

```
┌─────────────────────┐
│  unitree_sim_isaaclab│  (Isaac Sim仿真环境)
│                      │
│  - 运行仿真          │
│  - 提供相机图像      │──┐
│  - 提供机器人状态    │  │
└─────────────────────┘  │
                         │
                         │ DDS + ZMQ/HTTP
                         │
┌─────────────────────┐  │
│  vla_bridge_client  │  │ (桥接客户端，在unitree_sim_isaaclab环境中)
│                      │◄─┤
│  - 获取图像和状态   │  │
│  - 调用GR00T推理    │  │
│  - 发送控制命令     │  │
└─────────────────────┘  │
                         │
                         │ ZMQ/HTTP
                         │
┌─────────────────────┐  │
│gr00t_inference_server│◄─┤ (在Isaac-GR00T环境中)
└─────────────────────┘
│  - 加载GR00T模型    │
│  - 提供推理服务     │
│  - ZMQ或HTTP接口    │
└─────────────────────┘
```

## 部署步骤

### 步骤1: 启动GR00T推理服务

在**Isaac-GR00T虚拟环境**中运行：

```bash
cd Isaac-GR00T
# 激活Isaac-GR00T虚拟环境
# source activate gr00t_env  # 根据你的环境名称调整

# 使用ZMQ模式（默认，推荐）
python deployment_scripts/gr00t_inference_server.py \
    --checkpoint checkpoint-59900 \
    --port 5556 \
    --host 0.0.0.0 \
    --device cuda

# 或使用HTTP模式
python deployment_scripts/gr00t_inference_server.py \
    --checkpoint checkpoint-59900 \
    --port 8000 \
    --host 0.0.0.0 \
    --device cuda \
    --http-server
```

服务启动后会显示：
```
GR00T推理服务
============================================================
检查点路径: /path/to/checkpoint-59900
服务地址: 0.0.0.0:5556
计算设备: cuda
服务模式: ZMQ
============================================================
模型加载完成!
ZMQ服务器已启动，等待推理请求...
```

### 步骤2: 启动unitree_sim_isaaclab仿真环境

在**unitree_sim_isaaclab虚拟环境**中运行桥接客户端：

```bash
cd unitree_sim_isaaclab
# 激活unitree_sim_isaaclab虚拟环境
# source activate isaac_env  # 根据你的环境名称调整

# 运行桥接客户端（连接到GR00T服务）
python ../Isaac-GR00T/deployment_scripts/vla_bridge_client.py \
    --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
    --gr00t-host localhost \
    --gr00t-port 5556 \
    --gr00t-protocol zmq \
    --control-freq 30.0 \
    --enable-inspire-dds
```

如果GR00T服务使用HTTP模式：
```bash
python ../Isaac-GR00T/deployment_scripts/vla_bridge_client.py \
    --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
    --gr00t-host localhost \
    --gr00t-port 8000 \
    --gr00t-protocol http \
    --control-freq 30.0
```

### 步骤3: 可选 - 使用xr_teleoperate获取图像

如果需要使用`xr_teleoperate`的图像服务，可以：

1. **启动unitree_simbrella环境的图像服务器**：
   - unitree_sim_isaaclab已经内置了图像服务器，会自动发布图像

2. **在xr_teleoperate环境中获取图像**（如果需要）：
   - xr_teleoperate可以通过ZMQ客户端接收图像

## 通信协议

### ZMQ协议（推荐）

- **端口**: 默认5556（可配置）
- **优点**: 低延迟，高效
- **使用场景**: 本地或局域网部署

### HTTP协议

- **端口**: 默认8000（可配置）
- **优点**: 易于调试，跨网络
- **使用场景**: 跨网络部署，或需要REST API

## 参数说明

### GR00T推理服务参数

- `--checkpoint`: 模型检查点路径
- `--port`: 服务端口号
- `--host`: 服务监听地址（0.0.0.0表示所有网络接口）
- `--device`: 计算设备（cuda/cpu）
- `--denoising-steps`: 去噪步数（默认4）
- `--http-server`: 使用HTTP模式
- `--api-token`: API令牌（可选）

### 桥接客户端参数

- `--task`: Isaac Sim任务名称
- `--gr00t-host`: GR00T服务主机地址
- `--gr00t-port`: GR00T服务端口号
- `--gr00t-protocol`: 通信协议（zmq/http）
- `--gr00t-api-token`: GR00T服务API令牌
- `--control-freq`: 控制频率（Hz）
- `--enable-inspire-dds`: 启用Inspire手部DDS

## 网络配置

### 本地部署

如果三个程序在同一台机器上：
- GR00T服务: `localhost` 或 `127.0.0.1`
- 桥接客户端: `--gr00t-host localhost`

### 跨网络部署

如果GR00T服务在不同机器上：
- GR00T服务: `--host 0.0.0.0`（监听所有接口）
- 桥接客户端: `--gr00t-host <GR00T服务IP地址>`

**防火墙设置**：
- ZMQ: 确保端口5556（或自定义端口）开放
- HTTP: 确保端口8000（或自定义端口）开放

## 测试连接

### 测试GR00T服务（HTTP模式）

```bash
# 健康检查
curl http://localhost:8000/health

# 推理测试（需要有效的observation）
curl -X POST http://localhost:8000/act \
    -H "Content-Type: application/json" \
    -d '{"observation": {...}}'
```

### 测试GR00T服务（ZMQ模式）

可以使用Isaac-GR00T自带的客户端测试：
```bash
cd Isaac-GR00T
python scripts/inference_service.py --client \
    --host localhost \
    --port 5556
```

## 故障排除

### 连接失败

1. **检查GR00T服务是否启动**
   ```bash
   # 检查端口是否被占用
   netstat -an | grep 5556  # ZMQ
   netstat -an | grep 8000  # HTTP
   ```

2. **检查防火墙设置**
   - 确保端口未被阻止

3. **检查网络连接**
   ```bash
   # 测试TCP连接
   telnet <gr00t_host> <gr00t_port>
   ```

### 推理超时

1. **降低控制频率**
   ```bash
   --control-freq 10.0  # 降低到10Hz
   ```

2. **检查GPU可用性**
   - 确保GR00T服务使用GPU
   - 检查GPU内存是否足够

3. **增加超时时间**
   - 修改桥接客户端中的超时设置

### 图像获取失败

1. **检查环境配置**
   - 确保任务配置中包含`front_camera`
   - 检查`--enable-cameras`参数

2. **检查相机名称**
   - 确保环境中存在`front_camera`对象

## 性能优化

1. **降低控制频率**: 给推理留出足够时间
2. **使用ZMQ**: ZMQ比HTTP延迟更低
3. **GPU加速**: 确保GR00T服务使用GPU
4. **批量推理**: 如果有多个环境，可以考虑批量推理

## 下一步

- [ ] 添加图像压缩以减少网络传输
- [ ] 实现动作序列缓存，减少推理次数
- [ ] 添加监控和日志记录
- [ ] 支持多环境并行部署
- [ ] 集成xr_teleoperate的图像客户端

