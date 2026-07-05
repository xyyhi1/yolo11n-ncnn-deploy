#!/usr/bin/env bash
# 中文说明：该脚本负责配置并编译 C++ 工程。
# 中文说明：遇到错误立即退出、禁止使用未定义变量，并让管道返回真实错误。
set -euo pipefail

# 中文说明：取得脚本上一级目录的绝对路径，也就是项目根目录。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 中文说明：优先使用外部传入的 NCNN_DIR，否则使用本机默认安装路径。
NCNN_DIR="${NCNN_DIR:-$HOME/local/ncnn/lib/cmake/ncnn}"

# 中文说明：定义 CMake 配置命令数组，数组写法允许逐项添加中文解释。
configure_command=(
  # 中文说明：调用 cmake 程序。
  cmake
  # 中文说明：-S 后面指定源码根目录。
  -S "$ROOT"
  # 中文说明：-B 后面指定构建产物目录。
  -B "$ROOT/build"
  # 中文说明：使用 Release 模式编译，以获得优化后的性能。
  -DCMAKE_BUILD_TYPE=Release
  # 中文说明：告诉 CMake 在哪里寻找 ncnnConfig.cmake。
  -Dncnn_DIR="$NCNN_DIR"
)
# 中文说明：展开数组并执行 CMake 配置命令，双引号可避免路径被错误分词。
"${configure_command[@]}"
# 中文说明：调用全部 CPU 核心并行编译 build 目录中的工程。
cmake --build "$ROOT/build" -j"$(nproc)"

# 中文说明：在终端打印最终可执行文件的位置。
echo "Built: $ROOT/build/yolo_ncnn"
