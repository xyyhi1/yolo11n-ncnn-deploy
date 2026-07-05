// 文件作用：实现 NCNN 模型加载、Extractor 推理以及各阶段耗时统计。
// 阅读方式：每条有效代码的上一行都说明该代码的目的；空行用于分隔逻辑阶段。
// 中文说明：引入头文件 yolo_ncnn.h，使用其中声明的类型或函数。
#include "yolo_ncnn.h"

// 中文说明：引入头文件 algorithm，使用其中声明的类型或函数。
#include <algorithm>
// 中文说明：引入头文件 chrono，使用其中声明的类型或函数。
#include <chrono>
// 中文说明：引入头文件 iostream，使用其中声明的类型或函数。
#include <iostream>
// 中文说明：引入头文件 stdexcept，使用其中声明的类型或函数。
#include <stdexcept>
// 中文说明：引入头文件 utility，使用其中声明的类型或函数。
#include <utility>

// 中文说明：调用函数或构造对象，并传入括号中的参数。
bool YoloNcnnDetector::load(
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    const std::string& param_path,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    const std::string& bin_path,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    int num_threads,
    // 中文说明：声明变量 class_names)，用于保存当前处理结果。
    std::vector<std::string> class_names) {
    // 中文说明：更新变量或对象 num_threads_ 的值。
    num_threads_ = std::max(1, num_threads);
    // 中文说明：更新变量或对象 class_names_ 的值。
    class_names_ = std::move(class_names);
    // 中文说明：更新变量或对象 net_.opt.num_threads 的值。
    net_.opt.num_threads = num_threads_;
    // 中文说明：更新变量或对象 net_.opt.use_packing_layout 的值。
    net_.opt.use_packing_layout = true;
    // 中文说明：更新变量或对象 net_.opt.use_fp16_packed 的值。
    net_.opt.use_fp16_packed = false;
    // 中文说明：更新变量或对象 net_.opt.use_fp16_storage 的值。
    net_.opt.use_fp16_storage = false;
    // 中文说明：更新变量或对象 net_.opt.use_fp16_arithmetic 的值。
    net_.opt.use_fp16_arithmetic = false;

    // 中文说明：判断条件：(net_.load_param(param_path.c_str()) != 0) {
    if (net_.load_param(param_path.c_str()) != 0) {
        // 中文说明：把运行信息或错误信息输出到终端。
        std::cerr << "Failed to load NCNN param: " << param_path << '\n';
        // 中文说明：结束当前函数并返回结果：false;
        return false;
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：判断条件：(net_.load_model(bin_path.c_str()) != 0) {
    if (net_.load_model(bin_path.c_str()) != 0) {
        // 中文说明：把运行信息或错误信息输出到终端。
        std::cerr << "Failed to load NCNN bin: " << bin_path << '\n';
        // 中文说明：结束当前函数并返回结果：false;
        return false;
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：结束当前函数并返回结果：true;
    return true;
// 中文说明：结束当前函数、类型或代码块。
}

// 中文说明：定义检测函数，依次执行前处理、NCNN 推理、后处理并返回最终检测框。
std::vector<Detection> YoloNcnnDetector::detect(
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    const cv::Mat& image,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    int input_size,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    float conf_threshold,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    float iou_threshold,
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    DetectionTiming* timing) const {
    // 中文说明：为类型 Clock 定义更短的别名。
    using Clock = std::chrono::steady_clock;
    // 中文说明：计算结果并保存为只读变量 begin。
    const auto begin = Clock::now();
    // 中文说明：声明变量 preprocessed，用于保存当前处理结果。
    PreprocessResult preprocessed = preprocess_image(image, input_size);
    // 中文说明：计算结果并保存为只读变量 after_preprocess。
    const auto after_preprocess = Clock::now();

    // 中文说明：调用函数或构造对象，并传入括号中的参数。
    ncnn::Extractor extractor = net_.create_extractor();
    // 中文说明：调用函数或构造对象，并传入括号中的参数。
    extractor.set_light_mode(true);
    // 中文说明：判断条件：(extractor.input("in0", preprocessed.input) != 0) {
    if (extractor.input("in0", preprocessed.input) != 0) {
        // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        throw std::runtime_error("failed to bind NCNN input blob 'in0'");
    // 中文说明：结束当前函数、类型或代码块。
    }

    // 中文说明：声明变量 output，用于保存当前处理结果。
    ncnn::Mat output;
    // 中文说明：判断条件：(extractor.extract("out0", output) != 0) {
    if (extractor.extract("out0", output) != 0) {
        // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        throw std::runtime_error("failed to extract NCNN output blob 'out0'");
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：计算结果并保存为只读变量 after_inference。
    const auto after_inference = Clock::now();

    // 中文说明：声明变量 detections，用于保存当前处理结果。
    std::vector<Detection> detections = decode_yolo11_output(
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        output,
        // 中文说明：读取容器中的元素数量并参与当前计算或判断。
        static_cast<int>(class_names_.size()),
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        conf_threshold,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        iou_threshold,
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        preprocessed.letterbox);
    // 中文说明：计算结果并保存为只读变量 end。
    const auto end = Clock::now();

    // 中文说明：判断条件：(timing != nullptr) {
    if (timing != nullptr) {
        // 中文说明：计算结果并保存为只读变量 ms。
        const auto ms = [](auto a, auto b) {
            // 中文说明：结束当前函数并返回结果：std::chrono::duration<double, std::milli>(b - a).count();
            return std::chrono::duration<double, std::milli>(b - a).count();
        // 中文说明：结束当前函数、类型或代码块。
        };
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        timing->preprocess_ms = ms(begin, after_preprocess);
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        timing->inference_ms = ms(after_preprocess, after_inference);
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        timing->postprocess_ms = ms(after_inference, end);
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        timing->end_to_end_ms = ms(begin, end);
    // 中文说明：结束当前函数、类型或代码块。
    }

    // 中文说明：结束当前函数并返回结果：detections;
    return detections;
// 中文说明：结束当前函数、类型或代码块。
}
