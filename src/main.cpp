// 解析命令行参数，组织模型加载、推理、计时、结果保存和 CSV 输出。
#include "yolo_ncnn.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void print_usage() {
    std::cout
        << "Usage: yolo_ncnn --param model.param --bin model.bin --image input.jpg\n"
        << "                 --output result.jpg [--classes classes.txt] [--imgsz 640]\n"
        << "                 [--conf 0.25] [--iou 0.45] [--threads 4]\n"
        << "                 [--warmup 0] [--runs 1] [--benchmark-csv file.csv]\n";
}

// Parse command-line arguments and validate the --key value structure.
std::map<std::string, std::string> parse_arguments(int argc, char** argv) {
    std::map<std::string, std::string> arguments;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            print_usage();
            std::exit(0);
        }
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::runtime_error("invalid or incomplete argument: " + key);
        }
        arguments[key] = argv[++i];
    }
    return arguments;
}

std::string required(const std::map<std::string, std::string>& args, const std::string& key) {
    const auto it = args.find(key);
    if (it == args.end()) {
        throw std::runtime_error("missing required argument " + key);
    }
    return it->second;
}

std::string optional(
    const std::map<std::string, std::string>& args,
    const std::string& key,
    const std::string& fallback) {
    const auto it = args.find(key);
    return it == args.end() ? fallback : it->second;
}

std::vector<std::string> load_class_names(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open class names: " + path);
    }
    std::vector<std::string> names;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            names.push_back(line);
        }
    }
    if (names.empty()) {
        throw std::runtime_error("class names file is empty: " + path);
    }
    return names;
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double index = q * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(index);
    const std::size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = index - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double mean(const std::vector<double>& values) {
    return values.empty()
        ? 0.0
        : std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

// Export per-stage timing so benchmark statistics remain reproducible.
void write_csv(
    const std::string& path,
    int width,
    int height,
    int threads,
    const std::vector<DetectionTiming>& timings,
    std::size_t detection_count) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to write benchmark CSV: " + path);
    }
    output << "iteration,image_width,image_height,threads,preprocess_ms,inference_ms,postprocess_ms,end_to_end_ms,num_detections\n";
    for (std::size_t i = 0; i < timings.size(); ++i) {
        const auto& t = timings[i];
        output << i + 1 << ',' << width << ',' << height << ',' << threads << ','
               << t.preprocess_ms << ',' << t.inference_ms << ',' << t.postprocess_ms << ','
               << t.end_to_end_ms << ',' << detection_count << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_arguments(argc, argv);
        const std::string param_path = required(args, "--param");
        const std::string bin_path = required(args, "--bin");
        const std::string image_path = required(args, "--image");
        const std::string output_path = required(args, "--output");
        const std::string classes_path = optional(args, "--classes", "models/classes.txt");
        const int input_size = std::stoi(optional(args, "--imgsz", "640"));
        const float conf_threshold = std::stof(optional(args, "--conf", "0.25"));
        const float iou_threshold = std::stof(optional(args, "--iou", "0.45"));
        const int threads = std::stoi(optional(args, "--threads", "4"));
        const int warmup = std::stoi(optional(args, "--warmup", "0"));
        const int runs = std::max(1, std::stoi(optional(args, "--runs", "1")));
        const std::string csv_path = optional(args, "--benchmark-csv", "");

        for (const auto& path : {param_path, bin_path, image_path, classes_path}) {
            if (!fs::exists(path)) {
                throw std::runtime_error("file does not exist: " + path);
            }
        }

        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("OpenCV could not decode image: " + image_path);
        }
        const auto class_names = load_class_names(classes_path);

        YoloNcnnDetector detector;
        if (!detector.load(param_path, bin_path, threads, class_names)) {
            return 2;
        }

        std::cout << "Image: " << image.cols << 'x' << image.rows << '\n'
                  << "Input: 1x3x" << input_size << 'x' << input_size << " blob=in0\n"
                  << "Expected output: [1," << 4 + class_names.size() << ",8400] blob=out0\n"
                  << "Threads: " << threads << " warmup=" << warmup << " runs=" << runs << '\n';

        // Warm up the runtime before measurement to reduce cold-start noise.
        for (int i = 0; i < warmup; ++i) {
            detector.detect(image, input_size, conf_threshold, iou_threshold, nullptr);
        }

        std::vector<DetectionTiming> timings;
        timings.reserve(static_cast<std::size_t>(runs));
        std::vector<Detection> detections;
        for (int i = 0; i < runs; ++i) {
            DetectionTiming timing;
            detections = detector.detect(
                image, input_size, conf_threshold, iou_threshold, &timing);
            timings.push_back(timing);
        }

        cv::Mat rendered = image.clone();
        draw_detections(rendered, detections, detector.class_names());
        fs::create_directories(fs::path(output_path).parent_path());
        if (!cv::imwrite(output_path, rendered)) {
            throw std::runtime_error("failed to save result image: " + output_path);
        }

        std::vector<double> pre, infer, post, total;
        for (const auto& timing : timings) {
            pre.push_back(timing.preprocess_ms);
            infer.push_back(timing.inference_ms);
            post.push_back(timing.postprocess_ms);
            total.push_back(timing.end_to_end_ms);
        }
        const double total_mean = mean(total);
        std::cout << "Detections after NMS: " << detections.size() << '\n'
                  << "Mean preprocess: " << mean(pre) << " ms\n"
                  << "Mean inference: " << mean(infer) << " ms\n"
                  << "Mean postprocess: " << mean(post) << " ms\n"
                  << "End-to-end mean/p50/p95: " << total_mean << " / "
                  << percentile(total, 0.50) << " / " << percentile(total, 0.95) << " ms\n"
                  << "End-to-end FPS: " << (total_mean > 0.0 ? 1000.0 / total_mean : 0.0) << '\n'
                  << "Saved: " << output_path << '\n';

        if (!csv_path.empty()) {
            write_csv(csv_path, image.cols, image.rows, threads, timings, detections.size());
            std::cout << "Benchmark CSV: " << csv_path << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        print_usage();
        return 1;
    }
}
