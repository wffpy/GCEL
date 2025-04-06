#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 捕获错误的通用函数
check_error() {
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}❌ 错误：$1 失败${NC}"
        exit 1
    fi
}

# 获取工程根目录
project_dir="$(cd $(dirname $0)/.. && pwd)"
echo -e "${YELLOW}��� Project Directory: $project_dir${NC}"
cd "$project_dir" || exit 1

# 模式判断
if [[ "$#" -eq 0 ]]; then
    echo -e "${YELLOW}⚙️ 正在配置 Release 模式...${NC}"
    cmake -B build
    check_error "CMake Release 配置"
    echo -e "${GREEN}✅ Release 配置完成${NC}"

elif [[ "$1" == "-d" ]]; then
    echo -e "${YELLOW}⚙️ 正在配置 Debug 模式...${NC}"
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
    check_error "CMake Debug 配置"
    echo -e "${GREEN}✅ Debug 配置完成${NC}"

elif [[ "$1" == "-l" ]]; then
    echo -e "${YELLOW}��� 可构建目标如下:${NC}"
    cmake --build build --target help
    check_error "列出构建目标"

else
    echo -e "${RED}❌ 错误: 无效参数 '$1'${NC}"
    echo -e "${YELLOW}用法: $0 [-d] | [-l]${NC}"
    exit 1
fi

