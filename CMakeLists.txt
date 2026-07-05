# 文件作用：配置 C++17 工程，查找 OpenCV 和 NCNN，并生成 yolo_ncnn 可执行程序。
# 阅读方式：每条有效代码的上一行都说明该代码的目的；空行用于分隔逻辑阶段。
# 中文说明：声明构建该项目所需的最低 CMake 版本。
cmake_minimum_required(VERSION 3.16)
# 中文说明：定义项目名称，并指定项目使用 C++ 语言。
project(yolo11n_ncnn_deploy LANGUAGES CXX)

# 中文说明：把 C++ 语言标准设置为 C++17。
set(CMAKE_CXX_STANDARD 17)
# 中文说明：要求编译器必须支持指定的 C++ 标准。
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 中文说明：关闭编译器私有扩展，使用标准 C++ 语法。
set(CMAKE_CXX_EXTENSIONS OFF)

# 中文说明：查找并加载 OpenCV 的头文件、库和 CMake 配置。
find_package(OpenCV REQUIRED)
# 中文说明：查找并加载 ncnn 的头文件、库和 CMake 配置。
find_package(ncnn REQUIRED)

# 中文说明：开始定义可执行目标 yolo_ncnn 及其源文件列表。
add_executable(yolo_ncnn
    # 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    src/main.cpp
    # 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    src/yolo_ncnn.cpp
    # 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    src/preprocess.cpp
    # 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    src/postprocess.cpp
# 中文说明：结束当前跨行的 CMake 命令或函数调用。
)

# 中文说明：为可执行目标添加项目头文件和 OpenCV 头文件搜索目录。
target_include_directories(yolo_ncnn PRIVATE include ${OpenCV_INCLUDE_DIRS})
# 中文说明：把 NCNN 和 OpenCV 库链接到可执行程序。
target_link_libraries(yolo_ncnn PRIVATE ncnn ${OpenCV_LIBS})

# 中文说明：判断当前编译器是否为 GCC 或 Clang。
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # 中文说明：为当前编译器开启常用警告选项。
    target_compile_options(yolo_ncnn PRIVATE -Wall -Wextra -Wpedantic)
# 中文说明：结束 CMake 条件判断。
endif()
