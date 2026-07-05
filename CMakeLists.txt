# 配置 C++17 工程，查找 OpenCV 和 NCNN，并生成 yolo_ncnn 可执行程序。
cmake_minimum_required(VERSION 3.16)
project(yolo11n_ncnn_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable(yolo_ncnn
    src/main.cpp
    src/yolo_ncnn.cpp
    src/preprocess.cpp
    src/postprocess.cpp
)

target_include_directories(yolo_ncnn PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_ncnn PRIVATE ncnn ${OpenCV_LIBS})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(yolo_ncnn PRIVATE -Wall -Wextra -Wpedantic)
endif()
