// 实现 Letterbox、BGR 转 RGB、缩放、填充和像素归一化。
#include "yolo_ncnn.h"

#include <layer.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Preserve aspect ratio and retain the padding metadata required for box restoration.
PreprocessResult preprocess_image(const cv::Mat& image, int input_size) {
    if (image.empty()) {
        throw std::runtime_error("input image is empty");
    }
    if (input_size <= 0) {
        throw std::runtime_error("input size must be positive");
    }

    const int original_width = image.cols;
    const int original_height = image.rows;
    const float scale = std::min(
        static_cast<float>(input_size) / static_cast<float>(original_width),
        static_cast<float>(input_size) / static_cast<float>(original_height));

    const int resized_width = static_cast<int>(std::round(original_width * scale));
    const int resized_height = static_cast<int>(std::round(original_height * scale));
    const float dw = static_cast<float>(input_size - resized_width) / 2.0f;
    const float dh = static_cast<float>(input_size - resized_height) / 2.0f;

    // 与 Ultralytics 的 LetterBox 取整方式保持一致，避免上下或左右相差一个像素。
    const int pad_left = static_cast<int>(std::round(dw - 0.1f));
    const int pad_right = static_cast<int>(std::round(dw + 0.1f));
    const int pad_top = static_cast<int>(std::round(dh - 0.1f));
    const int pad_bottom = static_cast<int>(std::round(dh + 0.1f));

    // Convert OpenCV BGR input to the RGB layout used during model export.
    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(
        image.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        original_width,
        original_height,
        resized_width,
        resized_height);

    ncnn::Mat padded;
    ncnn::copy_make_border(
        resized,
        padded,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        ncnn::BORDER_CONSTANT,
        114.0f);

    // Match the training-time [0, 255] -> [0, 1] normalization exactly.
    const float normalizers[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    padded.substract_mean_normalize(nullptr, normalizers);

    PreprocessResult result;
    result.input = padded;
    result.letterbox = LetterboxInfo{
        scale,
        pad_left,
        pad_top,
        input_size,
        input_size,
        original_width,
        original_height};
    return result;
}
