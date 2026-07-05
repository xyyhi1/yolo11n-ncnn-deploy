// 解析 YOLO11 输出，过滤候选框，执行分类 NMS、坐标还原和画框。
#include "yolo_ncnn.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace {

float intersection_over_union(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float left = std::max(a.x, b.x);
    const float top = std::max(a.y, b.y);
    const float right = std::min(a.x + a.width, b.x + b.width);
    const float bottom = std::min(a.y + a.height, b.y + b.height);
    const float width = std::max(0.0f, right - left);
    const float height = std::max(0.0f, bottom - top);
    const float intersection = width * height;
    const float union_area = a.area() + b.area() - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

// Apply class-aware NMS so overlapping boxes from different classes do not suppress each other.
std::vector<Detection> class_aware_nms(
    std::vector<Detection> candidates,
    float iou_threshold) {
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    std::vector<Detection> kept;
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        kept.push_back(candidates[i]);
        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            if (!suppressed[j] && candidates[i].class_id == candidates[j].class_id &&
                intersection_over_union(candidates[i].box, candidates[j].box) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}

}  // namespace

// Decode either [4+C, N] or [N, 4+C] exports and restore boxes to the source image.
std::vector<Detection> decode_yolo11_output(
    const ncnn::Mat& output,
    int num_classes,
    float conf_threshold,
    float iou_threshold,
    const LetterboxInfo& letterbox) {
    if (num_classes <= 0) {
        throw std::runtime_error("class list is empty");
    }
    if (output.dims != 2) {
        throw std::runtime_error(
            "unsupported NCNN output dims=" + std::to_string(output.dims) +
            "; expected a 2D [4+C,N] or [N,4+C] tensor");
    }

    const int feature_count = 4 + num_classes;
    const bool feature_major = output.h == feature_count;
    const bool anchor_major = output.w == feature_count;
    if (!feature_major && !anchor_major) {
        throw std::runtime_error(
            "unexpected NCNN output shape w=" + std::to_string(output.w) +
            " h=" + std::to_string(output.h) +
            "; expected one axis to equal 4 + class_count = " + std::to_string(feature_count));
    }

    const int anchor_count = feature_major ? output.w : output.h;
    std::vector<Detection> candidates;
    candidates.reserve(static_cast<std::size_t>(anchor_count));

    for (int anchor = 0; anchor < anchor_count; ++anchor) {
        auto value_at = [&](int feature) -> float {
            return feature_major ? output.row(feature)[anchor] : output.row(anchor)[feature];
        };

        int best_class = -1;
        float best_score = -1.0f;
        for (int class_id = 0; class_id < num_classes; ++class_id) {
            const float score = value_at(4 + class_id);
            if (score > best_score) {
                best_score = score;
                best_class = class_id;
            }
        }
        if (best_score < conf_threshold) {
            continue;
        }

        const float center_x = value_at(0);
        const float center_y = value_at(1);
        const float width = value_at(2);
        const float height = value_at(3);

        // Remove letterbox padding before reversing the resize scale.
        float x1 = (center_x - width * 0.5f - letterbox.pad_left) / letterbox.scale;
        float y1 = (center_y - height * 0.5f - letterbox.pad_top) / letterbox.scale;
        float x2 = (center_x + width * 0.5f - letterbox.pad_left) / letterbox.scale;
        float y2 = (center_y + height * 0.5f - letterbox.pad_top) / letterbox.scale;

        x1 = std::clamp(x1, 0.0f, static_cast<float>(letterbox.original_width - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(letterbox.original_height - 1));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(letterbox.original_width - 1));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(letterbox.original_height - 1));

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }
        candidates.push_back(Detection{
            cv::Rect2f(x1, y1, x2 - x1, y2 - y1), best_class, best_score});
    }

    return class_aware_nms(std::move(candidates), iou_threshold);
}

void draw_detections(
    cv::Mat& image,
    const std::vector<Detection>& detections,
    const std::vector<std::string>& class_names) {
    static const cv::Scalar colors[] = {
        {255, 180, 0}, {0, 220, 255}, {80, 220, 80}, {255, 80, 180}, {180, 120, 255}};

    for (const Detection& detection : detections) {
        const cv::Scalar color = colors[detection.class_id % 5];
        cv::rectangle(image, detection.box, color, 2, cv::LINE_AA);

        std::ostringstream label;
        const std::string name = detection.class_id >= 0 &&
                detection.class_id < static_cast<int>(class_names.size())
            ? class_names[detection.class_id]
            : std::to_string(detection.class_id);
        label << name << ' ' << std::fixed << std::setprecision(2) << detection.confidence;

        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(
            label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        const int x = std::max(0, static_cast<int>(detection.box.x));
        const int y = std::max(text_size.height + 4, static_cast<int>(detection.box.y));
        cv::rectangle(
            image,
            cv::Rect(x, y - text_size.height - 4, text_size.width + 6, text_size.height + 6),
            color,
            cv::FILLED);
        cv::putText(
            image,
            label.str(),
            cv::Point(x + 3, y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(20, 20, 20),
            1,
            cv::LINE_AA);
    }
}
