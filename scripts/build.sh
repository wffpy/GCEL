#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
project_dir="$(cd $(dirname $0)/.. && pwd)"
echo -e "${YELLOW}��� 项目路径: $project_dir${NC}"
cd "$project_dir" || exit 1

# 检查命令是否成功执行
check_error() {
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}❌ 错误: $1 失败${NC}"
        exit 1
    fi
}

# 默认行为：编译
build_project() {
    local target=$1
    echo -e "${YELLOW}��� 正在编译项目...${NC}"
    if [ -z "$target" ]; then
        cmake --build build --target all
        check_error "编译所有目标"
    else
        cmake --build build --target "$target"
        check_error "编译目标 $target"
    fi
    echo -e "${GREEN}✅ 编译完成${NC}"
}

# 安装行为
install_project() {
    echo -e "${YELLOW}��� 正在安装项目...${NC}"
    cmake --build build --target install
    check_error "安装"
    echo -e "${GREEN}✅ 安装完成${NC}"
}

# 卸载行为
uninstall_project() {
    echo -e "${YELLOW}��� 正在卸载项目...${NC}"
    cmake --build build --target uninstall
    check_error "卸载"
    echo -e "${GREEN}✅ 卸载完成${NC}"
}

# 打包行为
package_project() {
    echo -e "${YELLOW}��� 正在打包项目...${NC}"
    cmake --build build --target package
    check_error "打包"
    echo -e "${GREEN}✅ 打包完成${NC}"
}

# 显示帮助信息
show_help() {
    echo -e "${YELLOW}用法: $0 [选项]${NC}"
    echo "选项:"
    echo "  -h, --help      显示帮助信息"
    echo "  -i, --install   安装项目"
    echo "  -u, --uninstall 卸载项目"
    echo "  -p, --package   打包项目"
    echo "  -t, --target    指定编译目标，不指定则编译所有目标"
    exit 0
}

# 解析参数
target=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
#        -i|--install)
#            install_project
#            exit 0
#            ;;
#        -u|--uninstall)
#            uninstall_project
#            exit 0
#            ;;
#        -p|--package)
#            package_project
#            exit 0
#            ;;
        -t|--target)
            if [ -z "$2" ]; then
                echo -e "${RED}❌ 错误: -t 参数需要一个目标名称${NC}"
                exit 1
            fi
            target="$2"
            shift
            ;;
        *)
            echo -e "${RED}❌ 错误: 无效参数 $1${NC}"
            show_help
            exit 1
            ;;
    esac
    shift
done

# 如果没有指定任何操作，或者只指定了 -t 参数，则执行编译
build_project "$target"

