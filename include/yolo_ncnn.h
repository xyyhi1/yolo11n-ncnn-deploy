// 声明推理所需的数据结构、前后处理函数以及 YOLO NCNN 检测器类。
// 防止这个头文件在同一编译单元中被重复包含。
#pragma once

#include <opencv2/core.hpp>

#include <net.h>

#include <string>
#include <vector>

struct LetterboxInfo {
    float scale = 1.0f;
    int pad_left = 0;
    int pad_top = 0;
    int input_width = 0;
    int input_height = 0;
    int original_width = 0;
    int original_height = 0;
};

struct Detection {
    cv::Rect2f box;
    int class_id = -1;
    float confidence = 0.0f;
};

struct DetectionTiming {
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double end_to_end_ms = 0.0;
};

struct PreprocessResult {
    ncnn::Mat input;
    LetterboxInfo letterbox;
};

// 声明图片前处理函数，具体实现位于 preprocess.cpp。
PreprocessResult preprocess_image(const cv::Mat& image, int input_size);

// 声明 YOLO11 输出解析函数，具体实现位于 postprocess.cpp。
std::vector<Detection> decode_yolo11_output(
    const ncnn::Mat& output,
    int num_classes,
    float conf_threshold,
    float iou_threshold,
    const LetterboxInfo& letterbox);

void draw_detections(
    cv::Mat& image,
    const std::vector<Detection>& detections,
    const std::vector<std::string>& class_names);

class YoloNcnnDetector {
public:
    bool load(
        const std::string& param_path,
        const std::string& bin_path,
        int num_threads,
        std::vector<std::string> class_names);

    // 声明单张图片检测接口，返回 NMS 后的检测结果。
    std::vector<Detection> detect(
        const cv::Mat& image,
        int input_size,
        float conf_threshold,
        float iou_threshold,
        DetectionTiming* timing = nullptr) const;

    const std::vector<std::string>& class_names() const { return class_names_; }
    int num_threads() const { return num_threads_; }

private:
    ncnn::Net net_;
    std::vector<std::string> class_names_;
    int num_threads_ = 1;
};
