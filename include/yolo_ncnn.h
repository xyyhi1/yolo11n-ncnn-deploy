// 文件作用：声明推理所需的数据结构、前后处理函数以及 YOLO NCNN 检测器类。
// 阅读方式：每条有效代码的上一行都说明该代码的目的；空行用于分隔逻辑阶段。
// 中文说明：防止这个头文件在同一编译单元中被重复包含。
#pragma once

// 中文说明：引入头文件 opencv2/core.hpp，使用其中声明的类型或函数。
#include <opencv2/core.hpp>

// 中文说明：引入头文件 net.h，使用其中声明的类型或函数。
#include <net.h>

// 中文说明：引入头文件 string，使用其中声明的类型或函数。
#include <string>
// 中文说明：引入头文件 vector，使用其中声明的类型或函数。
#include <vector>

// 中文说明：定义结构体 LetterboxInfo，集中保存一组相关数据。
struct LetterboxInfo {
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    float scale = 1.0f;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    int pad_left = 0;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    int pad_top = 0;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    int input_width = 0;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    int input_height = 0;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    int original_width = 0;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    int original_height = 0;
// 中文说明：结束当前函数、类型或代码块。
};

// 中文说明：定义结构体 Detection，集中保存一组相关数据。
struct Detection {
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    cv::Rect2f box;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    int class_id = -1;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    float confidence = 0.0f;
// 中文说明：结束当前函数、类型或代码块。
};

// 中文说明：定义结构体 DetectionTiming，集中保存一组相关数据。
struct DetectionTiming {
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    double preprocess_ms = 0.0;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    double inference_ms = 0.0;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    double postprocess_ms = 0.0;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    double end_to_end_ms = 0.0;
// 中文说明：结束当前函数、类型或代码块。
};

// 中文说明：定义结构体 PreprocessResult，集中保存一组相关数据。
struct PreprocessResult {
    // 中文说明：声明变量 input，用于保存当前处理结果。
    ncnn::Mat input;
    // 中文说明：声明变量 letterbox，用于保存当前处理结果。
    LetterboxInfo letterbox;
// 中文说明：结束当前函数、类型或代码块。
};

// 中文说明：声明图片前处理函数，具体实现位于 preprocess.cpp。
PreprocessResult preprocess_image(const cv::Mat& image, int input_size);

// 中文说明：声明 YOLO11 输出解析函数，具体实现位于 postprocess.cpp。
std::vector<Detection> decode_yolo11_output(
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    const ncnn::Mat& output,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    int num_classes,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    float conf_threshold,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    float iou_threshold,
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    const LetterboxInfo& letterbox);

// 中文说明：开始定义函数，并在括号中声明输入参数。
void draw_detections(
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    cv::Mat& image,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    const std::vector<Detection>& detections,
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    const std::vector<std::string>& class_names);

// 中文说明：定义类 YoloNcnnDetector {。
class YoloNcnnDetector {
// 中文说明：下面声明类的公开接口，类外代码可以调用。
public:
    // 中文说明：开始定义函数，并在括号中声明输入参数。
    bool load(
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        const std::string& param_path,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        const std::string& bin_path,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        int num_threads,
        // 中文说明：声明变量 class_names)，用于保存当前处理结果。
        std::vector<std::string> class_names);

    // 中文说明：声明单张图片检测接口，返回 NMS 后的检测结果。
    std::vector<Detection> detect(
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        const cv::Mat& image,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        int input_size,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        float conf_threshold,
        // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
        float iou_threshold,
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        DetectionTiming* timing = nullptr) const;

    // 中文说明：调用函数或构造对象，并传入括号中的参数。
    const std::vector<std::string>& class_names() const { return class_names_; }
    // 中文说明：开始定义函数，并在括号中声明输入参数。
    int num_threads() const { return num_threads_; }

// 中文说明：下面声明类的内部状态，类外代码不能直接访问。
private:
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    ncnn::Net net_;
    // 中文说明：声明变量 class_names_，用于保存当前处理结果。
    std::vector<std::string> class_names_;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    int num_threads_ = 1;
// 中文说明：结束当前函数、类型或代码块。
};
