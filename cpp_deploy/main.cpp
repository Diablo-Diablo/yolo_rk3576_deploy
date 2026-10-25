#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"

// ================= 配置区 =================
const char* MODEL_PATH = "/home/industio/yolo_test/yolo11n_rk3576_fp16.rknn";
const char* IMAGE_PATH = "/home/industio/yolo_test/bus.jpg";
const char* OUTPUT_PATH = "/home/industio/yolo_test/result_cpp.jpg";

const int MODEL_INPUT_W = 640;
const int MODEL_INPUT_H = 640;
const float CONF_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.45f;
// ==========================================

// 目标检测框结构体
struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
};

// 读取二进制模型文件辅助函数
static unsigned char* load_model(const char* filename, int* model_size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        std::cerr << "❌ 无法打开模型文件: " << filename << std::endl;
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char* data = (unsigned char*)malloc(size);
    if (!data) {
        fclose(fp);
        return nullptr;
    }
    fread(data, 1, size, fp);
    fclose(fp);
    *model_size = size;
    return data;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "🚀 RK3576 YOLOv11 原生 C++ NPU 部署实测" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. 读取原图
    cv::Mat orig_img = cv::imread(IMAGE_PATH);
    if (orig_img.empty()) {
        std::cerr << "❌ 无法加载图像: " << IMAGE_PATH << std::endl;
        return -1;
    }
    int orig_w = orig_img.cols;
    int orig_h = orig_img.rows;

    // 2. 图像预处理 (BGR -> RGB -> Resize 640x640)
    cv::Mat rgb_img;
    cv::cvtColor(orig_img, rgb_img, cv::COLOR_BGR2RGB);
    cv::Mat input_img;
    cv::resize(rgb_img, input_img, cv::Size(MODEL_INPUT_W, MODEL_INPUT_H));

    // 3. 加载 RKNN 二进制模型
    int model_data_size = 0;
    unsigned char* model_data = load_model(MODEL_PATH, &model_data_size);
    if (!model_data) return -1;

    // 4. 初始化 RKNN 上下文 (使用 RK3576 双核核心调度)
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    free(model_data); // 模型载入 NPU 内存后即可释放外部 buffer
    if (ret < 0) {
        std::cerr << "❌ rknn_init 失败, ret=" << ret << std::endl;
        return -1;
    }

    // 设置核心掩码为双核 (RK3576: NPU_CORE_0_1)
    ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1);
    if (ret < 0) {
        std::cout << "⚠️ 设置双核掩码失败，默认使用单核调度" << std::endl;
    } else {
        std::cout << "✓ 已启用 RK3576 双核 NPU 并发调度" << std::endl;
    }

    // 5. 设置输入张量 (NHWC, RGB, uint8)
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = MODEL_INPUT_W * MODEL_INPUT_H * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = input_img.data;

    ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        std::cerr << "❌ rknn_inputs_set 失败" << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    // 6. NPU 硬件推理 (精确打点测速)
    std::cout << "--> 正在执行 NPU 硬件推理..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    ret = rknn_run(ctx, NULL);
    auto t_end = std::chrono::high_resolution_clock::now();

    if (ret < 0) {
        std::cerr << "❌ rknn_run 失败" << std::endl;
        rknn_destroy(ctx);
        return -1;
    }
    double latency = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "✓ NPU 推理完成！C++ 单次硬件时延: " << latency << " ms" << std::endl;

    // 7. 获取输出张量 (由系统透明反量化为 float32)
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1; // 索取 float32 连续数据
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0) {
        std::cerr << "❌ rknn_outputs_get 失败" << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    // 8. 解析 (1, 84, 8400) 张量
    // 数据排布: 84 行, 8400 列
    // 前 4 行: cx, cy, w, h; 后 80 行: class scores
    float* out_data = (float*)outputs[0].buf;
    const int num_anchors = 8400;
    const int num_classes = 80;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    float scale_x = (float)orig_w / MODEL_INPUT_W;
    float scale_y = (float)orig_h / MODEL_INPUT_H;

    for (int i = 0; i < num_anchors; ++i) {
        // 寻找 80 个类别中的最高置信度
        float max_score = 0.0f;
        int max_cls_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = out_data[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                max_cls_id = c;
            }
        }

        if (max_score >= CONF_THRESHOLD) {
            float cx = out_data[0 * num_anchors + i];
            float cy = out_data[1 * num_anchors + i];
            float w  = out_data[2 * num_anchors + i];
            float h  = out_data[3 * num_anchors + i];

            // 还原到原图尺寸
            int left = std::max(0, (int)((cx - 0.5f * w) * scale_x));
            int top  = std::max(0, (int)((cy - 0.5f * h) * scale_y));
            int width  = std::min(orig_w - left, (int)(w * scale_x));
            int height = std::min(orig_h - top,  (int)(h * scale_y));

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(max_score);
            class_ids.push_back(max_cls_id);
        }
    }

    // 9. 执行 OpenCV NMS 过滤重叠候选框
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);
    std::cout << "✓ 检测目标总数 (NMS 过滤后): " << indices.size() << " 个" << std::endl;

    // 10. 在图像上画框标注
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        float conf = confidences[idx];
        int cls_id = class_ids[idx];

        cv::rectangle(orig_img, box, cv::Scalar(0, 255, 0), 2);
        std::string label = "Cls " + std::to_string(cls_id) + ": " + cv::format("%.2f", conf);
        cv::putText(orig_img, label, cv::Point(box.x, std::max(box.y - 5, 15)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

        std::cout << "  - 检测到目标: 类别=" << cls_id << ", 置信度=" << conf 
                  << ", 框=[" << box.x << ", " << box.y << ", " << box.width << ", " << box.height << "]" << std::endl;
    }

    // 11. 保存结果图像
    cv::imwrite(OUTPUT_PATH, orig_img);
    std::cout << "✓ 结果图片已保存至: " << OUTPUT_PATH << std::endl;

    // 12. 释放资源
    rknn_outputs_release(ctx, 1, outputs);
    rknn_destroy(ctx);

    std::cout << "========================================" << std::endl;
    std::cout << "🎉 端到端 C++ NPU 部署全流程验证成功！" << std::endl;
    return 0;
}

