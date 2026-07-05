// 文件作用：解析 YOLO11 输出，过滤候选框，执行分类 NMS、坐标还原和画框。
// 阅读方式：每条有效代码的上一行都说明该代码的目的；空行用于分隔逻辑阶段。
// 中文说明：引入头文件 yolo_ncnn.h，使用其中声明的类型或函数。
#include "yolo_ncnn.h"

// 中文说明：引入头文件 opencv2/imgproc.hpp，使用其中声明的类型或函数。
#include <opencv2/imgproc.hpp>

// 中文说明：引入头文件 algorithm，使用其中声明的类型或函数。
#include <algorithm>
// 中文说明：引入头文件 cmath，使用其中声明的类型或函数。
#include <cmath>
// 中文说明：引入头文件 iomanip，使用其中声明的类型或函数。
#include <iomanip>
// 中文说明：引入头文件 numeric，使用其中声明的类型或函数。
#include <numeric>
// 中文说明：引入头文件 sstream，使用其中声明的类型或函数。
#include <sstream>
// 中文说明：引入头文件 stdexcept，使用其中声明的类型或函数。
#include <stdexcept>

// 中文说明：进入匿名命名空间，使内部辅助函数只在本源文件可见。
namespace {

// 中文说明：开始定义函数，并在括号中声明输入参数。
float intersection_over_union(const cv::Rect2f& a, const cv::Rect2f& b) {
    // 中文说明：计算结果并保存为只读变量 left。
    const float left = std::max(a.x, b.x);
    // 中文说明：计算结果并保存为只读变量 top。
    const float top = std::max(a.y, b.y);
    // 中文说明：计算结果并保存为只读变量 right。
    const float right = std::min(a.x + a.width, b.x + b.width);
    // 中文说明：计算结果并保存为只读变量 bottom。
    const float bottom = std::min(a.y + a.height, b.y + b.height);
    // 中文说明：计算结果并保存为只读变量 width。
    const float width = std::max(0.0f, right - left);
    // 中文说明：计算结果并保存为只读变量 height。
    const float height = std::max(0.0f, bottom - top);
    // 中文说明：计算结果并保存为只读变量 intersection。
    const float intersection = width * height;
    // 中文说明：计算结果并保存为只读变量 union_area。
    const float union_area = a.area() + b.area() - intersection;
    // 中文说明：结束当前函数并返回结果：union_area > 0.0f ? intersection / union_area : 0.0f;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
// 中文说明：结束当前函数、类型或代码块。
}

// 中文说明：定义分类 NMS 函数，同类别高重叠框会被抑制，不同类别之间互不影响。
std::vector<Detection> class_aware_nms(
    // 中文说明：声明变量 candidates,，用于保存当前处理结果。
    std::vector<Detection> candidates,
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    float iou_threshold) {
    // 中文说明：调用函数或构造对象，并传入括号中的参数。
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        // 中文说明：结束当前函数并返回结果：a.confidence > b.confidence;
        return a.confidence > b.confidence;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    });

    // 中文说明：声明变量 kept，用于保存当前处理结果。
    std::vector<Detection> kept;
    // 中文说明：为每个候选框建立抑制标记，初始状态全部为 false。
    std::vector<bool> suppressed(candidates.size(), false);
    // 中文说明：按顺序循环处理：(std::size_t i = 0; i < candidates.size(); ++i) {
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        // 中文说明：判断条件：(suppressed[i]) {
        if (suppressed[i]) {
            // 中文说明：跳过本次循环剩余代码，直接进入下一次循环。
            continue;
        // 中文说明：结束当前函数、类型或代码块。
        }
        // 中文说明：把当前元素追加到动态数组末尾。
        kept.push_back(candidates[i]);
        // 中文说明：按顺序循环处理：(std::size_t j = i + 1; j < candidates.size(); ++j) {
        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            // 中文说明：判断条件：(!suppressed[j] && candidates[i].class_id == candidates[j].class_id &&
            if (!suppressed[j] && candidates[i].class_id == candidates[j].class_id &&
                // 中文说明：调用函数或构造对象，并传入括号中的参数。
                intersection_over_union(candidates[i].box, candidates[j].box) > iou_threshold) {
                // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
                suppressed[j] = true;
            // 中文说明：结束当前函数、类型或代码块。
            }
        // 中文说明：结束当前函数、类型或代码块。
        }
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：结束当前函数并返回结果：kept;
    return kept;
// 中文说明：结束当前函数、类型或代码块。
}

// 中文说明：结束匿名命名空间。
}  // namespace

// 中文说明：定义 YOLO11 输出解析函数，把原始张量转换为还原到原图坐标的检测框。
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
    const LetterboxInfo& letterbox) {
    // 中文说明：判断条件：(num_classes <= 0) {
    if (num_classes <= 0) {
        // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        throw std::runtime_error("class list is empty");
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：判断条件：(output.dims != 2) {
    if (output.dims != 2) {
        // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        throw std::runtime_error(
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            "unsupported NCNN output dims=" + std::to_string(output.dims) +
            // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
            "; expected a 2D [4+C,N] or [N,4+C] tensor");
    // 中文说明：结束当前函数、类型或代码块。
    }

    // 中文说明：计算结果并保存为只读变量 feature_count。
    const int feature_count = 4 + num_classes;
    // 中文说明：计算结果并保存为只读变量 feature_major。
    const bool feature_major = output.h == feature_count;
    // 中文说明：计算结果并保存为只读变量 anchor_major。
    const bool anchor_major = output.w == feature_count;
    // 中文说明：判断条件：(!feature_major && !anchor_major) {
    if (!feature_major && !anchor_major) {
        // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        throw std::runtime_error(
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            "unexpected NCNN output shape w=" + std::to_string(output.w) +
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            " h=" + std::to_string(output.h) +
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            "; expected one axis to equal 4 + class_count = " + std::to_string(feature_count));
    // 中文说明：结束当前函数、类型或代码块。
    }

    // 中文说明：计算结果并保存为只读变量 anchor_count。
    const int anchor_count = feature_major ? output.w : output.h;
    // 中文说明：声明变量 candidates，用于保存当前处理结果。
    std::vector<Detection> candidates;
    // 中文说明：提前预留容器容量，减少循环中的重复内存分配。
    candidates.reserve(static_cast<std::size_t>(anchor_count));

    // 中文说明：按顺序循环处理：(int anchor = 0; anchor < anchor_count; ++anchor) {
    for (int anchor = 0; anchor < anchor_count; ++anchor) {
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        auto value_at = [&](int feature) -> float {
            // 中文说明：结束当前函数并返回结果：feature_major ? output.row(feature)[anchor] : output.row(anchor)[feature];
            return feature_major ? output.row(feature)[anchor] : output.row(anchor)[feature];
        // 中文说明：结束当前函数、类型或代码块。
        };

        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        int best_class = -1;
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        float best_score = -1.0f;
        // 中文说明：按顺序循环处理：(int class_id = 0; class_id < num_classes; ++class_id) {
        for (int class_id = 0; class_id < num_classes; ++class_id) {
            // 中文说明：计算结果并保存为只读变量 score。
            const float score = value_at(4 + class_id);
            // 中文说明：判断条件：(score > best_score) {
            if (score > best_score) {
                // 中文说明：更新变量或对象 best_score 的值。
                best_score = score;
                // 中文说明：更新变量或对象 best_class 的值。
                best_class = class_id;
            // 中文说明：结束当前函数、类型或代码块。
            }
        // 中文说明：结束当前函数、类型或代码块。
        }
        // 中文说明：判断条件：(best_score < conf_threshold) {
        if (best_score < conf_threshold) {
            // 中文说明：跳过本次循环剩余代码，直接进入下一次循环。
            continue;
        // 中文说明：结束当前函数、类型或代码块。
        }

        // 中文说明：计算结果并保存为只读变量 center_x。
        const float center_x = value_at(0);
        // 中文说明：计算结果并保存为只读变量 center_y。
        const float center_y = value_at(1);
        // 中文说明：计算结果并保存为只读变量 width。
        const float width = value_at(2);
        // 中文说明：计算结果并保存为只读变量 height。
        const float height = value_at(3);

        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        float x1 = (center_x - width * 0.5f - letterbox.pad_left) / letterbox.scale;
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        float y1 = (center_y - height * 0.5f - letterbox.pad_top) / letterbox.scale;
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        float x2 = (center_x + width * 0.5f - letterbox.pad_left) / letterbox.scale;
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        float y2 = (center_y + height * 0.5f - letterbox.pad_top) / letterbox.scale;

        // 中文说明：更新变量或对象 x1 的值。
        x1 = std::clamp(x1, 0.0f, static_cast<float>(letterbox.original_width - 1));
        // 中文说明：更新变量或对象 y1 的值。
        y1 = std::clamp(y1, 0.0f, static_cast<float>(letterbox.original_height - 1));
        // 中文说明：更新变量或对象 x2 的值。
        x2 = std::clamp(x2, 0.0f, static_cast<float>(letterbox.original_width - 1));
        // 中文说明：更新变量或对象 y2 的值。
        y2 = std::clamp(y2, 0.0f, static_cast<float>(letterbox.original_height - 1));

        // 中文说明：判断条件：(x2 <= x1 || y2 <= y1) {
        if (x2 <= x1 || y2 <= y1) {
            // 中文说明：跳过本次循环剩余代码，直接进入下一次循环。
            continue;
        // 中文说明：结束当前函数、类型或代码块。
        }
        // 中文说明：把当前元素追加到动态数组末尾。
        candidates.push_back(Detection{
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            cv::Rect2f(x1, y1, x2 - x1, y2 - y1), best_class, best_score});
    // 中文说明：结束当前函数、类型或代码块。
    }

    // 中文说明：结束当前函数并返回结果：class_aware_nms(std::move(candidates), iou_threshold);
    return class_aware_nms(std::move(candidates), iou_threshold);
// 中文说明：结束当前函数、类型或代码块。
}

// 中文说明：开始定义函数，并在括号中声明输入参数。
void draw_detections(
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    cv::Mat& image,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    const std::vector<Detection>& detections,
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    const std::vector<std::string>& class_names) {
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    static const cv::Scalar colors[] = {
        // 中文说明：进入当前函数、类型或代码块。
        {255, 180, 0}, {0, 220, 255}, {80, 220, 80}, {255, 80, 180}, {180, 120, 255}};

    // 中文说明：按顺序循环处理：(const Detection& detection : detections) {
    for (const Detection& detection : detections) {
        // 中文说明：计算结果并保存为只读变量 color。
        const cv::Scalar color = colors[detection.class_id % 5];
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        cv::rectangle(image, detection.box, color, 2, cv::LINE_AA);

        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        std::ostringstream label;
        // 中文说明：计算结果并保存为只读变量 name。
        const std::string name = detection.class_id >= 0 &&
                // 中文说明：读取容器中的元素数量并参与当前计算或判断。
                detection.class_id < static_cast<int>(class_names.size())
            // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
            ? class_names[detection.class_id]
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            : std::to_string(detection.class_id);
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        label << name << ' ' << std::fixed << std::setprecision(2) << detection.confidence;

        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        int baseline = 0;
        // 中文说明：计算结果并保存为只读变量 text_size。
        const cv::Size text_size = cv::getTextSize(
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        // 中文说明：计算结果并保存为只读变量 x。
        const int x = std::max(0, static_cast<int>(detection.box.x));
        // 中文说明：计算结果并保存为只读变量 y。
        const int y = std::max(text_size.height + 4, static_cast<int>(detection.box.y));
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        cv::rectangle(
            // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
            image,
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            cv::Rect(x, y - text_size.height - 4, text_size.width + 6, text_size.height + 6),
            // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
            color,
            // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
            cv::FILLED);
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        cv::putText(
            // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
            image,
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            label.str(),
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            cv::Point(x + 3, y),
            // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
            cv::FONT_HERSHEY_SIMPLEX,
            // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
            0.45,
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            cv::Scalar(20, 20, 20),
            // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
            1,
            // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
            cv::LINE_AA);
    // 中文说明：结束当前函数、类型或代码块。
    }
// 中文说明：结束当前函数、类型或代码块。
}
