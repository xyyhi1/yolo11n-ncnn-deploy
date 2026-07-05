// 文件作用：解析命令行参数，组织模型加载、推理、计时、结果保存和 CSV 输出。
// 阅读方式：每条有效代码的上一行都说明该代码的目的；空行用于分隔逻辑阶段。
// 中文说明：引入头文件 yolo_ncnn.h，使用其中声明的类型或函数。
#include "yolo_ncnn.h"

// 中文说明：引入头文件 opencv2/imgcodecs.hpp，使用其中声明的类型或函数。
#include <opencv2/imgcodecs.hpp>

// 中文说明：引入头文件 algorithm，使用其中声明的类型或函数。
#include <algorithm>
// 中文说明：引入头文件 filesystem，使用其中声明的类型或函数。
#include <filesystem>
// 中文说明：引入头文件 fstream，使用其中声明的类型或函数。
#include <fstream>
// 中文说明：引入头文件 iostream，使用其中声明的类型或函数。
#include <iostream>
// 中文说明：引入头文件 map，使用其中声明的类型或函数。
#include <map>
// 中文说明：引入头文件 numeric，使用其中声明的类型或函数。
#include <numeric>
// 中文说明：引入头文件 stdexcept，使用其中声明的类型或函数。
#include <stdexcept>
// 中文说明：引入头文件 string，使用其中声明的类型或函数。
#include <string>
// 中文说明：引入头文件 vector，使用其中声明的类型或函数。
#include <vector>

// 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
namespace fs = std::filesystem;

// 中文说明：进入匿名命名空间，使内部辅助函数只在本源文件可见。
namespace {

// 中文说明：开始定义函数，并在括号中声明输入参数。
void print_usage() {
    // 中文说明：把运行信息或错误信息输出到终端。
    std::cout
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        << "Usage: yolo_ncnn --param model.param --bin model.bin --image input.jpg\n"
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        << "                 --output result.jpg [--classes classes.txt] [--imgsz 640]\n"
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        << "                 [--conf 0.25] [--iou 0.45] [--threads 4]\n"
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        << "                 [--warmup 0] [--runs 1] [--benchmark-csv file.csv]\n";
// 中文说明：结束当前函数、类型或代码块。
}

// 中文说明：调用函数或构造对象，并传入括号中的参数。
std::map<std::string, std::string> parse_arguments(int argc, char** argv) {
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    std::map<std::string, std::string> arguments;
    // 中文说明：按顺序循环处理：(int i = 1; i < argc; ++i) {
    for (int i = 1; i < argc; ++i) {
        // 中文说明：计算结果并保存为只读变量 key。
        const std::string key = argv[i];
        // 中文说明：判断条件：(key == "--help" || key == "-h") {
        if (key == "--help" || key == "-h") {
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            print_usage();
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            std::exit(0);
        // 中文说明：结束当前函数、类型或代码块。
        }
        // 中文说明：判断条件：(key.rfind("--", 0) != 0 || i + 1 >= argc) {
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
            throw std::runtime_error("invalid or incomplete argument: " + key);
        // 中文说明：结束当前函数、类型或代码块。
        }
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        arguments[key] = argv[++i];
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：结束当前函数并返回结果：arguments;
    return arguments;
// 中文说明：结束当前函数、类型或代码块。
}

// 中文说明：开始定义函数，并在括号中声明输入参数。
std::string required(const std::map<std::string, std::string>& args, const std::string& key) {
    // 中文说明：计算结果并保存为只读变量 it。
    const auto it = args.find(key);
    // 中文说明：判断条件：(it == args.end()) {
    if (it == args.end()) {
        // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        throw std::runtime_error("missing required argument " + key);
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：结束当前函数并返回结果：it->second;
    return it->second;
// 中文说明：结束当前函数、类型或代码块。
}

// 中文说明：开始定义函数，并在括号中声明输入参数。
std::string optional(
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    const std::map<std::string, std::string>& args,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    const std::string& key,
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    const std::string& fallback) {
    // 中文说明：计算结果并保存为只读变量 it。
    const auto it = args.find(key);
    // 中文说明：结束当前函数并返回结果：it == args.end() ? fallback : it->second;
    return it == args.end() ? fallback : it->second;
// 中文说明：结束当前函数、类型或代码块。
}

// 中文说明：定义类别文件读取函数，每个非空行对应一个类别名称。
std::vector<std::string> load_class_names(const std::string& path) {
    // 中文说明：开始定义函数，并在括号中声明输入参数。
    std::ifstream input(path);
    // 中文说明：判断条件：(!input) {
    if (!input) {
        // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        throw std::runtime_error("failed to open class names: " + path);
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：声明变量 names，用于保存当前处理结果。
    std::vector<std::string> names;
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    std::string line;
    // 中文说明：条件成立时重复执行循环：(std::getline(input, line)) {
    while (std::getline(input, line)) {
        // 中文说明：判断条件：(!line.empty()) {
        if (!line.empty()) {
            // 中文说明：把当前元素追加到动态数组末尾。
            names.push_back(line);
        // 中文说明：结束当前函数、类型或代码块。
        }
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：判断条件：(names.empty()) {
    if (names.empty()) {
        // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        throw std::runtime_error("class names file is empty: " + path);
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：结束当前函数并返回结果：names;
    return names;
// 中文说明：结束当前函数、类型或代码块。
}

// 中文说明：开始定义函数，并在括号中声明输入参数。
double percentile(std::vector<double> values, double q) {
    // 中文说明：判断条件：(values.empty()) {
    if (values.empty()) {
        // 中文说明：结束当前函数并返回结果：0.0;
        return 0.0;
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：调用函数或构造对象，并传入括号中的参数。
    std::sort(values.begin(), values.end());
    // 中文说明：计算结果并保存为只读变量 index。
    const double index = q * static_cast<double>(values.size() - 1);
    // 中文说明：计算结果并保存为只读变量 lower。
    const std::size_t lower = static_cast<std::size_t>(index);
    // 中文说明：计算结果并保存为只读变量 upper。
    const std::size_t upper = std::min(lower + 1, values.size() - 1);
    // 中文说明：计算结果并保存为只读变量 fraction。
    const double fraction = index - static_cast<double>(lower);
    // 中文说明：结束当前函数并返回结果：values[lower] * (1.0 - fraction) + values[upper] * fraction;
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
// 中文说明：结束当前函数、类型或代码块。
}

// 中文说明：开始定义函数，并在括号中声明输入参数。
double mean(const std::vector<double>& values) {
    // 中文说明：结束当前函数并返回结果：values.empty()
    return values.empty()
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        ? 0.0
        // 中文说明：读取容器中的元素数量并参与当前计算或判断。
        : std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
// 中文说明：结束当前函数、类型或代码块。
}

// 中文说明：开始定义函数，并在括号中声明输入参数。
void write_csv(
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    const std::string& path,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    int width,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    int height,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    int threads,
    // 中文说明：继续提供当前跨行函数、列表或结构体的下一个参数。
    const std::vector<DetectionTiming>& timings,
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    std::size_t detection_count) {
    // 中文说明：调用函数或构造对象，并传入括号中的参数。
    fs::create_directories(fs::path(path).parent_path());
    // 中文说明：开始定义函数，并在括号中声明输入参数。
    std::ofstream output(path);
    // 中文说明：判断条件：(!output) {
    if (!output) {
        // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
        throw std::runtime_error("failed to write benchmark CSV: " + path);
    // 中文说明：结束当前函数、类型或代码块。
    }
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    output << "iteration,image_width,image_height,threads,preprocess_ms,inference_ms,postprocess_ms,end_to_end_ms,num_detections\n";
    // 中文说明：按顺序循环处理：(std::size_t i = 0; i < timings.size(); ++i) {
    for (std::size_t i = 0; i < timings.size(); ++i) {
        // 中文说明：计算结果并保存为只读变量 t。
        const auto& t = timings[i];
        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        output << i + 1 << ',' << width << ',' << height << ',' << threads << ','
               // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
               << t.preprocess_ms << ',' << t.inference_ms << ',' << t.postprocess_ms << ','
               // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
               << t.end_to_end_ms << ',' << detection_count << '\n';
    // 中文说明：结束当前函数、类型或代码块。
    }
// 中文说明：结束当前函数、类型或代码块。
}

// 中文说明：结束匿名命名空间。
}  // namespace

// 中文说明：开始定义函数，并在括号中声明输入参数。
int main(int argc, char** argv) {
    // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
    try {
        // 中文说明：计算结果并保存为只读变量 args。
        const auto args = parse_arguments(argc, argv);
        // 中文说明：计算结果并保存为只读变量 param_path。
        const std::string param_path = required(args, "--param");
        // 中文说明：计算结果并保存为只读变量 bin_path。
        const std::string bin_path = required(args, "--bin");
        // 中文说明：计算结果并保存为只读变量 image_path。
        const std::string image_path = required(args, "--image");
        // 中文说明：计算结果并保存为只读变量 output_path。
        const std::string output_path = required(args, "--output");
        // 中文说明：计算结果并保存为只读变量 classes_path。
        const std::string classes_path = optional(args, "--classes", "models/classes.txt");
        // 中文说明：计算结果并保存为只读变量 input_size。
        const int input_size = std::stoi(optional(args, "--imgsz", "640"));
        // 中文说明：计算结果并保存为只读变量 conf_threshold。
        const float conf_threshold = std::stof(optional(args, "--conf", "0.25"));
        // 中文说明：计算结果并保存为只读变量 iou_threshold。
        const float iou_threshold = std::stof(optional(args, "--iou", "0.45"));
        // 中文说明：计算结果并保存为只读变量 threads。
        const int threads = std::stoi(optional(args, "--threads", "4"));
        // 中文说明：计算结果并保存为只读变量 warmup。
        const int warmup = std::stoi(optional(args, "--warmup", "0"));
        // 中文说明：计算结果并保存为只读变量 runs。
        const int runs = std::max(1, std::stoi(optional(args, "--runs", "1")));
        // 中文说明：计算结果并保存为只读变量 csv_path。
        const std::string csv_path = optional(args, "--benchmark-csv", "");

        // 中文说明：按顺序循环处理：(const auto& path : {param_path, bin_path, image_path, classes_path}) {
        for (const auto& path : {param_path, bin_path, image_path, classes_path}) {
            // 中文说明：判断条件：(!fs::exists(path)) {
            if (!fs::exists(path)) {
                // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
                throw std::runtime_error("file does not exist: " + path);
            // 中文说明：结束当前函数、类型或代码块。
            }
        // 中文说明：结束当前函数、类型或代码块。
        }

        // 中文说明：声明变量 image，用于保存当前处理结果。
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        // 中文说明：判断条件：(image.empty()) {
        if (image.empty()) {
            // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
            throw std::runtime_error("OpenCV could not decode image: " + image_path);
        // 中文说明：结束当前函数、类型或代码块。
        }
        // 中文说明：计算结果并保存为只读变量 class_names。
        const auto class_names = load_class_names(classes_path);

        // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
        YoloNcnnDetector detector;
        // 中文说明：判断条件：(!detector.load(param_path, bin_path, threads, class_names)) {
        if (!detector.load(param_path, bin_path, threads, class_names)) {
            // 中文说明：结束当前函数并返回结果：2;
            return 2;
        // 中文说明：结束当前函数、类型或代码块。
        }

        // 中文说明：把运行信息或错误信息输出到终端。
        std::cout << "Image: " << image.cols << 'x' << image.rows << '\n'
                  // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
                  << "Input: 1x3x" << input_size << 'x' << input_size << " blob=in0\n"
                  // 中文说明：读取容器中的元素数量并参与当前计算或判断。
                  << "Expected output: [1," << 4 + class_names.size() << ",8400] blob=out0\n"
                  // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
                  << "Threads: " << threads << " warmup=" << warmup << " runs=" << runs << '\n';

        // 中文说明：按顺序循环处理：(int i = 0; i < warmup; ++i) {
        for (int i = 0; i < warmup; ++i) {
            // 中文说明：调用函数或构造对象，并传入括号中的参数。
            detector.detect(image, input_size, conf_threshold, iou_threshold, nullptr);
        // 中文说明：结束当前函数、类型或代码块。
        }

        // 中文说明：声明变量 timings，用于保存当前处理结果。
        std::vector<DetectionTiming> timings;
        // 中文说明：提前预留容器容量，减少循环中的重复内存分配。
        timings.reserve(static_cast<std::size_t>(runs));
        // 中文说明：声明变量 detections，用于保存当前处理结果。
        std::vector<Detection> detections;
        // 中文说明：按顺序循环处理：(int i = 0; i < runs; ++i) {
        for (int i = 0; i < runs; ++i) {
            // 中文说明：声明变量 timing，用于保存当前处理结果。
            DetectionTiming timing;
            // 中文说明：更新变量或对象 detections 的值。
            detections = detector.detect(
                // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
                image, input_size, conf_threshold, iou_threshold, &timing);
            // 中文说明：把当前元素追加到动态数组末尾。
            timings.push_back(timing);
        // 中文说明：结束当前函数、类型或代码块。
        }

        // 中文说明：声明变量 rendered，用于保存当前处理结果。
        cv::Mat rendered = image.clone();
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        draw_detections(rendered, detections, detector.class_names());
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        fs::create_directories(fs::path(output_path).parent_path());
        // 中文说明：判断条件：(!cv::imwrite(output_path, rendered)) {
        if (!cv::imwrite(output_path, rendered)) {
            // 中文说明：发现非法状态，抛出异常并停止当前处理流程。
            throw std::runtime_error("failed to save result image: " + output_path);
        // 中文说明：结束当前函数、类型或代码块。
        }

        // 中文说明：声明变量 pre,，用于保存当前处理结果。
        std::vector<double> pre, infer, post, total;
        // 中文说明：按顺序循环处理：(const auto& timing : timings) {
        for (const auto& timing : timings) {
            // 中文说明：把当前元素追加到动态数组末尾。
            pre.push_back(timing.preprocess_ms);
            // 中文说明：把当前元素追加到动态数组末尾。
            infer.push_back(timing.inference_ms);
            // 中文说明：把当前元素追加到动态数组末尾。
            post.push_back(timing.postprocess_ms);
            // 中文说明：把当前元素追加到动态数组末尾。
            total.push_back(timing.end_to_end_ms);
        // 中文说明：结束当前函数、类型或代码块。
        }
        // 中文说明：计算结果并保存为只读变量 total_mean。
        const double total_mean = mean(total);
        // 中文说明：把运行信息或错误信息输出到终端。
        std::cout << "Detections after NMS: " << detections.size() << '\n'
                  // 中文说明：调用函数或构造对象，并传入括号中的参数。
                  << "Mean preprocess: " << mean(pre) << " ms\n"
                  // 中文说明：调用函数或构造对象，并传入括号中的参数。
                  << "Mean inference: " << mean(infer) << " ms\n"
                  // 中文说明：调用函数或构造对象，并传入括号中的参数。
                  << "Mean postprocess: " << mean(post) << " ms\n"
                  // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
                  << "End-to-end mean/p50/p95: " << total_mean << " / "
                  // 中文说明：调用函数或构造对象，并传入括号中的参数。
                  << percentile(total, 0.50) << " / " << percentile(total, 0.95) << " ms\n"
                  // 中文说明：调用函数或构造对象，并传入括号中的参数。
                  << "End-to-end FPS: " << (total_mean > 0.0 ? 1000.0 / total_mean : 0.0) << '\n'
                  // 中文说明：执行当前语句；其输入和结果由所在函数的上下文决定。
                  << "Saved: " << output_path << '\n';

        // 中文说明：判断条件：(!csv_path.empty()) {
        if (!csv_path.empty()) {
            // 中文说明：读取容器中的元素数量并参与当前计算或判断。
            write_csv(csv_path, image.cols, image.rows, threads, timings, detections.size());
            // 中文说明：把运行信息或错误信息输出到终端。
            std::cout << "Benchmark CSV: " << csv_path << '\n';
        // 中文说明：结束当前函数、类型或代码块。
        }
        // 中文说明：结束当前函数并返回结果：0;
        return 0;
    // 中文说明：调用函数或构造对象，并传入括号中的参数。
    } catch (const std::exception& error) {
        // 中文说明：把运行信息或错误信息输出到终端。
        std::cerr << "ERROR: " << error.what() << '\n';
        // 中文说明：调用函数或构造对象，并传入括号中的参数。
        print_usage();
        // 中文说明：结束当前函数并返回结果：1;
        return 1;
    // 中文说明：结束当前函数、类型或代码块。
    }
// 中文说明：结束当前函数、类型或代码块。
}
