// 实现 NCNN 模型加载、Extractor 推理以及各阶段耗时统计。
#include "yolo_ncnn.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <utility>

bool YoloNcnnDetector::load(
    const std::string& param_path,
    const std::string& bin_path,
    int num_threads,
    std::vector<std::string> class_names) {
    num_threads_ = std::max(1, num_threads);
    class_names_ = std::move(class_names);
    net_.opt.num_threads = num_threads_;
    net_.opt.use_packing_layout = true;
    // Keep the CPU benchmark comparable by disabling implicit FP16 execution paths.
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_storage = false;
    net_.opt.use_fp16_arithmetic = false;

    if (net_.load_param(param_path.c_str()) != 0) {
        std::cerr << "Failed to load NCNN param: " << param_path << '\n';
        return false;
    }
    if (net_.load_model(bin_path.c_str()) != 0) {
        std::cerr << "Failed to load NCNN bin: " << bin_path << '\n';
        return false;
    }
    return true;
}

// 定义检测函数，依次执行前处理、NCNN 推理、后处理并返回最终检测框。
std::vector<Detection> YoloNcnnDetector::detect(
    const cv::Mat& image,
    int input_size,
    float conf_threshold,
    float iou_threshold,
    DetectionTiming* timing) const {
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    PreprocessResult preprocessed = preprocess_image(image, input_size);
    const auto after_preprocess = Clock::now();

    ncnn::Extractor extractor = net_.create_extractor();
    extractor.set_light_mode(true);
    if (extractor.input("in0", preprocessed.input) != 0) {
        throw std::runtime_error("failed to bind NCNN input blob 'in0'");
    }

    ncnn::Mat output;
    if (extractor.extract("out0", output) != 0) {
        throw std::runtime_error("failed to extract NCNN output blob 'out0'");
    }
    const auto after_inference = Clock::now();

    std::vector<Detection> detections = decode_yolo11_output(
        output,
        static_cast<int>(class_names_.size()),
        conf_threshold,
        iou_threshold,
        preprocessed.letterbox);
    const auto end = Clock::now();

    if (timing != nullptr) {
        const auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        timing->preprocess_ms = ms(begin, after_preprocess);
        timing->inference_ms = ms(after_preprocess, after_inference);
        timing->postprocess_ms = ms(after_inference, end);
        timing->end_to_end_ms = ms(begin, end);
    }

    return detections;
}
