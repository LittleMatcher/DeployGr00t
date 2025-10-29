#!/bin/bash
# 分布式VLA部署启动脚本
# 此脚本帮助启动GR00T推理服务和桥接客户端

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}分布式VLA部署启动脚本${NC}"
echo -e "${GREEN}============================================================${NC}"

# 默认参数
CHECKPOINT="checkpoint-59900"
GR00T_PORT=5556
GR00T_HOST="0.0.0.0"
GR00T_PROTOCOL="zmq"
TASK="Isaac-PickPlace-Cylinder-G129-Inspire-Joint"
CONTROL_FREQ=30.0
DEVICE="cuda"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --gr00t-port)
            GR00T_PORT="$2"
            shift 2
            ;;
        --gr00t-host)
            GR00T_HOST="$2"
            shift 2
            ;;
        --gr00t-protocol)
            GR00T_PROTOCOL="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --control-freq)
            CONTROL_FREQ="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --http)
            GR00T_PROTOCOL="http"
            GR00T_PORT=8000
            shift
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --checkpoint PATH        模型检查点路径 (默认: checkpoint-59900)"
            echo "  --gr00t-port PORT        GR00T服务端口 (默认: 5556)"
            echo "  --gr00t-host HOST        GR00T服务主机 (默认: 0.0.0.0)"
            echo "  --gr00t-protocol PROTO   通信协议 zmq/http (默认: zmq)"
            echo "  --task TASK              Isaac Sim任务名称"
            echo "  --control-freq FREQ      控制频率 (默认: 30.0)"
            echo "  --device DEVICE          计算设备 cuda/cpu (默认: cuda)"
            echo "  --http                   使用HTTP协议 (端口改为8000)"
            echo ""
            echo "示例:"
            echo "  $0 --checkpoint checkpoint-59900 --http"
            echo "  $0 --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint --control-freq 20"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 获取脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GR00T_DIR="$(dirname "$SCRIPT_DIR")"
ISAAC_DIR="$(dirname "$GR00T_DIR")/unitree_sim_isaaclab"

echo -e "${YELLOW}配置信息:${NC}"
echo "  检查点: $CHECKPOINT"
echo "  GR00T服务: $GR00T_PROTOCOL://$GR00T_HOST:$GR00T_PORT"
echo "  任务: $TASK"
echo "  控制频率: $CONTROL_FREQ Hz"
echo ""

# 检查目录
if [ ! -d "$GR00T_DIR" ]; then
    echo -e "${RED}错误: Isaac-GR00T目录不存在: $GR00T_DIR${NC}"
    exit 1
fi

if [ ! -d "$ISAAC_DIR" ]; then
    echo -e "${RED}错误: unitree_sim_isaaclab目录不存在: $ISAAC_DIR${NC}"
    exit 1
fi

# 创建日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo -e "${GREEN}步骤1: 启动GR00T推理服务${NC}"
echo "请在Isaac-GR00T虚拟环境中运行以下命令："
echo ""
echo -e "${YELLOW}cd $GR00T_DIR${NC}"
if [ "$GR00T_PROTOCOL" == "http" ]; then
    echo -e "${YELLOW}python deployment_scripts/gr00t_inference_server.py \\${NC}"
    echo -e "${YELLOW}    --checkpoint $CHECKPOINT \\${NC}"
    echo -e "${YELLOW}    --port $GR00T_PORT \\${NC}"
    echo -e "${YELLOW}    --host $GR00T_HOST \\${NC}"
    echo -e "${YELLOW}    --device $DEVICE \\${NC}"
    echo -e "${YELLOW}    --http-server${NC}"
else
    echo -e "${YELLOW}python deployment_scripts/gr00t_inference_server.py \\${NC}"
    echo -e "${YELLOW}    --checkpoint $CHECKPOINT \\${NC}"
    echo -e "${YELLOW}    --port $GR00T_PORT \\${NC}"
    echo -e "${YELLOW}    --host $GR00T_HOST \\${NC}"
    echo -e "${YELLOW}    --device $DEVICE${NC}"
fi
echo ""
echo -e "${YELLOW}或直接运行:${NC}"
if [ "$GR00T_PROTOCOL" == "http" ]; then
    echo -e "${YELLOW}python $SCRIPT_DIR/gr00t_inference_server.py --checkpoint $CHECKPOINT --port $GR00T_PORT --host $GR00T_HOST --device $DEVICE --http-server${NC}"
else
    echo -e "${YELLOW}python $SCRIPT_DIR/gr00t_inference_server.py --checkpoint $CHECKPOINT --port $GR00T_PORT --host $GR00T_HOST --device $DEVICE${NC}"
fi
echo ""
echo -e "${GREEN}步骤2: 启动桥接客户端${NC}"
echo "请在unitree_sim_isaaclab虚拟环境中运行以下命令："
echo ""
echo -e "${YELLOW}cd $ISAAC_DIR${NC}"
echo -e "${YELLOW}python $SCRIPT_DIR/vla_bridge_client.py \\${NC}"
echo -e "${YELLOW}    --task $TASK \\${NC}"
echo -e "${YELLOW}    --gr00t-host localhost \\${NC}"
echo -e "${YELLOW}    --gr00t-port $GR00T_PORT \\${NC}"
echo -e "${YELLOW}    --gr00t-protocol $GR00T_PROTOCOL \\${NC}"
echo -e "${YELLOW}    --control-freq $CONTROL_FREQ${NC}"
echo ""

# 提供交互式选项
read -p "是否现在启动GR00T服务? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}启动GR00T推理服务...${NC}"
    cd "$GR00T_DIR"
    if [ "$GR00T_PROTOCOL" == "http" ]; then
        python deployment_scripts/gr00t_inference_server.py \
            --checkpoint "$CHECKPOINT" \
            --port "$GR00T_PORT" \
            --host "$GR00T_HOST" \
            --device "$DEVICE" \
            --http-server \
            2>&1 | tee "$LOG_DIR/gr00t_server.log" &
    else
        python deployment_scripts/gr00t_inference_server.py \
            --checkpoint "$CHECKPOINT" \
            --port "$GR00T_PORT" \
            --host "$GR00T_HOST" \
            --device "$DEVICE" \
            2>&1 | tee "$LOG_DIR/gr00t_server.log" &
    fi
    GR00T_PID=$!
    echo -e "${GREEN}GR00T服务已启动 (PID: $GR00T_PID)${NC}"
    echo "日志文件: $LOG_DIR/gr00t_server.log"
    
    # 等待服务启动
    echo -e "${YELLOW}等待GR00T服务启动...${NC}"
    sleep 5
    
    read -p "是否现在启动桥接客户端? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}启动桥接客户端...${NC}"
        cd "$ISAAC_DIR"
        python "$SCRIPT_DIR/vla_bridge_client.py" \
            --task "$TASK" \
            --gr00t-host localhost \
            --gr00t-port "$GR00T_PORT" \
            --gr00t-protocol "$GR00T_PROTOCOL" \
            --control-freq "$CONTROL_FREQ" \
            2>&1 | tee "$LOG_DIR/bridge_client.log"
    else
        echo -e "${YELLOW}可以稍后手动启动桥接客户端${NC}"
        echo "按 Ctrl+C 停止GR00T服务"
        wait $GR00T_PID
    fi
else
    echo -e "${YELLOW}请按照上面的命令手动启动服务${NC}"
fi

