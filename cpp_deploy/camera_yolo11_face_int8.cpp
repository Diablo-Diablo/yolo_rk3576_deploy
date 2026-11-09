#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"

// ================= 配置区 =================
const char* MODEL_PATH = "/home/industio/yolo_test/yolo11n_face_split_int8.rknn";
const char* CAMERA_DEV = "/dev/video42";
const int CAM_WIDTH = 1920;
const int CAM_HEIGHT = 1080;

const int MODEL_INPUT_W = 640;
const int MODEL_INPUT_H = 640;
const float CONF_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.45f;
// ==========================================

// 原生 V4L2 mmap 缓冲区结构体
struct Buffer {
    void* start;
    size_t length;
};

class V4L2Camera {
public:
    int fd = -1;
    std::vector<Buffer> buffers;

    bool init(const char* dev_name, int width, int height) {
        fd = open(dev_name, O_RDWR);
        if (fd < 0) {
            std::cerr << "❌ 打开摄像头设备失败: " << dev_name << std::endl;
            return false;
        }

        // 1. 设置视频格式 (Multiplanar NV12)
        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        fmt.fmt.pix_mp.width = width;
        fmt.fmt.pix_mp.height = height;
        fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
        fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
        fmt.fmt.pix_mp.num_planes = 1;

        if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
            std::cerr << "❌ 设置格式失败 (VIDIOC_S_FMT)" << std::endl;
            close(fd);
            return false;
        }

        // 2. 申请 3 个内核缓冲区
        struct v4l2_requestbuffers req;
        memset(&req, 0, sizeof(req));
        req.count = 3;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        req.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
            std::cerr << "❌ 申请缓冲区失败 (VIDIOC_REQBUFS)" << std::endl;
            close(fd);
            return false;
        }

        // 3. mmap 映射内存到用户空间
        buffers.resize(req.count);
        for (unsigned int i = 0; i < req.count; ++i) {
            struct v4l2_plane planes[1];
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            memset(planes, 0, sizeof(planes));

            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            buf.m.planes = planes;
            buf.length = 1;

            if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
                std::cerr << "❌ 查询缓冲区失败 (VIDIOC_QUERYBUF)" << std::endl;
                return false;
            }

            buffers[i].length = buf.m.planes[0].length;
            buffers[i].start = mmap(NULL, buf.m.planes[0].length,
                                   PROT_READ | PROT_WRITE, MAP_SHARED,
                                   fd, buf.m.planes[0].m.mem_offset);

            if (buffers[i].start == MAP_FAILED) {
                std::cerr << "❌ mmap 内存映射失败" << std::endl;
                return false;
            }

            // 放入输入队列
            if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
                std::cerr << "❌ 放回缓冲区队列失败 (VIDIOC_QBUF)" << std::endl;
                return false;
            }
        }

        // 4. 启动视频流采集
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
            std::cerr << "❌ 启动视频流失败 (VIDIOC_STREAMON)" << std::endl;
            return false;
        }

        std::cout << "✓ 原生 V4L2 硬件多平面流启动成功" << std::endl;
        return true;
    }

    bool get_frame(cv::Mat& bgr_img) {
        struct v4l2_plane planes[1];
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.m.planes = planes;
        buf.length = 1;

        // 从队列中取出填满数据的缓冲区
        if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
            return false;
        }

        // NV12 转换到 BGR: (height * 1.5, width) 单通道内存包装
        cv::Mat nv12_mat(CAM_HEIGHT * 3 / 2, CAM_WIDTH, CV_8UC1, buffers[buf.index].start);
        cv::cvtColor(nv12_mat, bgr_img, cv::COLOR_YUV2BGR_NV12);

        // 处理完将缓冲区重新排队回内核
        ioctl(fd, VIDIOC_QBUF, &buf);
        return true;
    }

    void release() {
        if (fd >= 0) {
            enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            ioctl(fd, VIDIOC_STREAMOFF, &type);
            for (auto& b : buffers) {
                if (b.start != MAP_FAILED) munmap(b.start, b.length);
            }
            close(fd);
            fd = -1;
        }
    }

    ~V4L2Camera() { release(); }
};

static unsigned char* load_model(const char* filename, int* model_size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return nullptr;
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char* data = (unsigned char*)malloc(size);
    if (!data) {
        fclose(fp);
        return nullptr;
    }
    if (fread(data, 1, size, fp) != (size_t)size) {}
    fclose(fp);
    *model_size = size;
    return data;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "🚀 RK3576 原生 V4L2 摄像头 YOLOv11 实时推理" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. 初始化原生 V4L2 摄像头
    V4L2Camera camera;
    if (!camera.init(CAMERA_DEV, CAM_WIDTH, CAM_HEIGHT)) {
        return -1;
    }

    // 2. 初始化 NPU 运行环境
    int model_data_size = 0;
    unsigned char* model_data = load_model(MODEL_PATH, &model_data_size);
    if (!model_data) return -1;

    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    free(model_data);
    if (ret < 0) return -1;

    ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1);
    if (ret < 0) {
        std::cerr << "rknn_set_core_mask failed: " << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    // Split outputs: identify boxes/scores by shape, not output order.
    rknn_input_output_num io_num = {};
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0 || io_num.n_input != 1 || io_num.n_output != 2) {
        std::cerr << "Expected one input and two split outputs" << std::endl;
        rknn_destroy(ctx);
        return -1;
    }
    const int num_anchors = 8400;
    int boxes_index = -1;
    int scores_index = -1;
    bool boxes_channel_first = true;
    for (unsigned int i = 0; i < io_num.n_output; ++i) {
        rknn_tensor_attr attr = {};
        attr.index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret < 0) {
            std::cerr << "Output attribute query failed: " << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }
        std::cout << "Output " << i << " (" << attr.name << "):";
        for (unsigned int d = 0; d < attr.n_dims; ++d) {
            std::cout << " " << attr.dims[d];
        }
        std::cout << "; scale=" << attr.scale << "; zp=" << attr.zp
                  << std::endl;
        if (attr.n_dims != 3 || attr.dims[0] != 1) continue;
        if (attr.n_elems == 4 * num_anchors &&
            ((attr.dims[1] == 4 && attr.dims[2] == num_anchors) ||
             (attr.dims[1] == num_anchors && attr.dims[2] == 4))) {
            boxes_index = i;
            boxes_channel_first = attr.dims[1] == 4;
        } else if (attr.n_elems == num_anchors &&
                   ((attr.dims[1] == 1 && attr.dims[2] == num_anchors) ||
                    (attr.dims[1] == num_anchors && attr.dims[2] == 1))) {
            scores_index = i;
        }
    }
    if (boxes_index < 0 || scores_index < 0) {
        std::cerr << "Unsupported split output shapes" << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    // 3. 准备缓冲区结构
    cv::Mat frame, rgb_frame, input_img;
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = MODEL_INPUT_W * MODEL_INPUT_H * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    rknn_output outputs[2];
    memset(outputs, 0, sizeof(outputs));
    for (unsigned int i = 0; i < 2; ++i) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }

    const std::string window_name = "Purple Pi OH2 - YOLOv11 Live";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    std::cout << "--> 开始实时屏幕渲染 (按键盘 ESC 退出)..." << std::endl;

    int frame_count = 0;
    double fps_display = 0.0;

    bool inference_failed = false;
    while (true) {
        auto t0 = std::chrono::steady_clock::now();

        // 原生 V4L2 取流
        if (!camera.get_frame(frame)) {
            continue;
        }

        int orig_w = frame.cols;
        int orig_h = frame.rows;
        float scale_x = (float)orig_w / MODEL_INPUT_W;
        float scale_y = (float)orig_h / MODEL_INPUT_H;

        // 预处理 (Resize 640x640, BGR -> RGB)
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        cv::resize(rgb_frame, input_img, cv::Size(MODEL_INPUT_W, MODEL_INPUT_H));
        inputs[0].buf = input_img.data;

        // NPU 推理
        ret = rknn_inputs_set(ctx, 1, inputs);
        if (ret == RKNN_SUCC) ret = rknn_run(ctx, NULL);
        if (ret == RKNN_SUCC) ret = rknn_outputs_get(ctx, 2, outputs, NULL);
        if (ret != RKNN_SUCC) {
            std::cerr << "RKNN inference/output retrieval failed: " << ret
                      << std::endl;
            inference_failed = true;
            break;
        }
        if (!outputs[boxes_index].buf || !outputs[scores_index].buf ||
            outputs[boxes_index].size < 4 * num_anchors * sizeof(float) ||
            outputs[scores_index].size < num_anchors * sizeof(float)) {
            std::cerr << "Invalid split output buffers" << std::endl;
            rknn_outputs_release(ctx, 2, outputs);
            inference_failed = true;
            break;
        }

        // 解析输出
        const float* box_data = (const float*)outputs[boxes_index].buf;
        const float* score_data = (const float*)outputs[scores_index].buf;
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        for (int i = 0; i < num_anchors; ++i) {
            // Scores are already probabilities; do not apply sigmoid again.
            float max_score = score_data[i];
            int max_cls_id = 0;

            if (max_score >= CONF_THRESHOLD) {
                auto coordinate = [&](int c) {
                    return box_data[boxes_channel_first ? c * num_anchors + i
                                                       : i * 4 + c];
                };
                float cx = coordinate(0);
                float cy = coordinate(1);
                float w = coordinate(2);
                float h = coordinate(3);
                if (!std::isfinite(max_score) || !std::isfinite(cx) ||
                    !std::isfinite(cy) || !std::isfinite(w) ||
                    !std::isfinite(h) || w <= 0 || h <= 0) continue;

                int left = (int)std::clamp((cx - 0.5f * w) * scale_x,
                                           0.0f, (float)orig_w);
                int top = (int)std::clamp((cy - 0.5f * h) * scale_y,
                                          0.0f, (float)orig_h);
                int right = (int)std::clamp((cx + 0.5f * w) * scale_x,
                                            0.0f, (float)orig_w);
                int bottom = (int)std::clamp((cy + 0.5f * h) * scale_y,
                                             0.0f, (float)orig_h);
                int width = right - left;
                int height = bottom - top;
                if (width <= 0 || height <= 0) continue;

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(max_score);
                class_ids.push_back(max_cls_id);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

        for (int idx : indices) {
            cv::Rect box = boxes[idx];
            float conf = confidences[idx];
            int cls_id = class_ids[idx];

            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            std::string label = "ID:" + std::to_string(cls_id) + " " + cv::format("%.2f", conf);
            cv::putText(frame, label, cv::Point(box.x, std::max(box.y - 5, 15)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        ret = rknn_outputs_release(ctx, 2, outputs);
        if (ret != RKNN_SUCC) {
            std::cerr << "Output release failed: " << ret << std::endl;
            inference_failed = true;
            break;
        }

        auto t1 = std::chrono::steady_clock::now();
        double frame_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        frame_count++;
        if (frame_count % 5 == 0) {
            fps_display = 1000.0 / frame_ms;
        }

        std::string fps_text = "FPS: " + cv::format("%.1f", fps_display) + " | Latency: " + cv::format("%.1f", frame_ms) + "ms";
        cv::putText(frame, fps_text, cv::Point(20, 45), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        cv::imshow(window_name, frame);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    rknn_destroy(ctx);
    camera.release();
    return inference_failed ? -1 : 0;
}
