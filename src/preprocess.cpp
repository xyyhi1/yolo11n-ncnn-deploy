// 文件作用：实现 Letterbox、BGR 转 RGB、缩放、填充和像素归一化。
// 阅读方式：每条有效代码的上一行都说明该代码的目的；空行用于分隔逻辑阶段。
// 中文说明：引入头文件 yolo_ncnn.h，使用其中声明的类型或函数。
#include "yolo_ncnn.h"

// 中文说明：引入头文件 layer.h，使用其中声明的类型或函数。
#include <layer.h>

// 中文说明：引入头文件 algorithm，使用其中声明的类型或函数。
#include <algorithm>
// 中文说明：引入头文件 cmath，使用其中声明的类型或函数。
#include <cmath>
// 中文说明：引入头文件 stdexcept，使用其中声明的类型或函数。
#include <stdexcept>

// 中文说明：定义图片前处理函数，输入原图和模型尺寸，返回 NCNN 输入张量及 Letterbox 参数。
PreprocessResult preprocess_image(const cv::Mat& image, int input_size) {
    // 中文说明：判断条件：(image.empty()) {
    if (image.empty()) {
        // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        throw std::runtime_error("input image is empty");
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：判断条件：(input_size <= 0) {
    if (input_size <= 0) {
        // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        throw std::runtime_error("input size must be positive");
    // 中文说明：结束当前函数、类型或代码块。
    }

    // 中文说明：计算结果并保存为只读变量 original_width。
    const int original_width = image.cols;
    // 中文说明：计算结果并保存为只读变量 original_height。
    const int original_height = image.rows;
    // 中文说明：计算结果并保存为只读变量 scale。
    const float scale = std::min(
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        static_cast<float>(input_size) / static_cast<float>(original_width),
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        static_cast<float>(input_size) / static_cast<float>(original_height));

    // 中文说明：计算结果并保存为只读变量 resized_width。
    const int resized_width = static_cast<int>(std::round(original_width * scale));
    // 中文说明：计算结果并保存为只读变量 resized_height。
    const int resized_height = static_cast<int>(std::round(original_height * scale));
    // 中文说明：计算结果并保存为只读变量 dw。
    const float dw = static_cast<float>(input_size - resized_width) / 2.0f;
    // 中文说明：计算结果并保存为只读变量 dh。
    const float dh = static_cast<float>(input_size - resized_height) / 2.0f;

    // 与 Ultralytics 的 LetterBox 取整方式保持一致，避免上下或左右相差一个像素。
    // 中文说明：计算结果并保存为只读变量 pad_left。
    const int pad_left = static_cast<int>(std::round(dw - 0.1f));
    // 中文说明：计算结果并保存为只读变量 pad_right。
    const int pad_right = static_cast<int>(std::round(dw + 0.1f));
    // 中文说明：计算结果并保存为只读变量 pad_top。
    const int pad_top = static_cast<int>(std::round(dh - 0.1f));
    // 中文说明：计算结果并保存为只读变量 pad_bottom。
    const int pad_bottom = static_cast<int>(std::round(dh + 0.1f));

    // 中文说明：声明变量 resized，用于保存当前处理结果。
    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        image.data,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        ncnn::Mat::PIXEL_BGR2RGB,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        original_width,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        original_height,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        resized_width,
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        resized_height);

    // 中文说明：声明变量 padded，用于保存当前处理结果。
    ncnn::Mat padded;
    // 中文说明：调用函数或构造对象，并传入括号中的参数。
    ncnn::copy_make_border(
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        resized,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        padded,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        pad_top,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        pad_bottom,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        pad_left,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        pad_right,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        ncnn::BORDER_CONSTANT,
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        114.0f);

    // 中文说明：计算结果并保存为只读变量 normalizers[3]。
    const float normalizers[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    // 中文说明：调用函数或构造对象，并传入括号中的参数。
    padded.substract_mean_normalize(nullptr, normalizers);

    // 中文说明：声明变量 result，用于保存当前处理结果。
    PreprocessResult result;
    // 中文说明：更新变量或对象 result.input 的值。
    result.input = padded;
    // 中文说明：更新变量或对象 result.letterbox 的值。
    result.letterbox = LetterboxInfo{
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        scale,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        pad_left,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        pad_top,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        input_size,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        input_size,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        original_width,
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        original_height};
    // 中文说明：结束当前函数并返回结果：result;
    return result;
// 中文说明：结束当前函数、类型或代码块。
}
