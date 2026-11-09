// Staged RK3576 face recognition. No landmark alignment or tracking.
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include <cerrno>
#include <cctype>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Vector = std::array<float, 512>;
static double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
static void check(int ret, const char* operation) {
    require(ret == RKNN_SUCC, std::string(operation)+": "+std::to_string(ret));
}
static Vector normalize(Vector v) {
    double sum = 0;
    for (float x : v) {
        require(std::isfinite(x), "Non-finite embedding");
        sum += double(x)*x;
    }
    require(sum > 1e-24, "Zero embedding norm");
    for (auto& x : v) x /= std::sqrt(sum);
    return v;
}
// Read only unpickled, little-endian float NPY vectors with one identity.
static Vector load_reference(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    require(bool(f), "Cannot open reference: "+path.string());
    char magic[6]; f.read(magic, 6);
    require(f && std::memcmp(magic, "\x93NUMPY", 6)==0, "Invalid NPY magic");
    unsigned char version[2]; f.read(reinterpret_cast<char*>(version), 2);
    require(version[0]>=1 && version[0]<=3, "Unsupported NPY version");
    unsigned char len[4] = {};
    int n = version[0]==1 ? 2 : 4;
    f.read(reinterpret_cast<char*>(len), n);
    uint32_t length = uint32_t(len[0]) | uint32_t(len[1])<<8 |
                      uint32_t(len[2])<<16 | uint32_t(len[3])<<24;
    require(f && length<=65536, "Invalid NPY header size");
    std::string header(length, '\0'); f.read(header.data(), length);
    std::smatch match;
    require(std::regex_search(header, match,
            std::regex("['\"]descr['\"]\\s*:\\s*['\"]([<=>|]f[48])['\"]")),
            "Reference must be float32/float64 NPY");
    std::string dtype = match[1];
    require(dtype[0]!='>', "Big-endian NPY is unsupported");
    require(std::regex_search(header, match,
            std::regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)")),
            "Missing NPY shape");
    std::string shape = match[1];
    shape.erase(std::remove_if(shape.begin(),shape.end(),
                [](unsigned char c){return std::isspace(c);}),shape.end());
    require(shape=="512," || shape=="1,512" || shape=="1,512,",
            "Reference shape must be (512,) or (1,512)");
    Vector v{};
    for (float& x : v) {
        if (dtype.back()=='4') f.read(reinterpret_cast<char*>(&x), 4);
        else { double d; f.read(reinterpret_cast<char*>(&d), 8); x=float(d); }
    }
    require(bool(f), "Truncated NPY payload");
    require(f.peek()==std::char_traits<char>::eof(), "Unexpected NPY payload");
    return normalize(v);
}
static void save_embeddings(const fs::path& path,
                            const std::vector<Vector>& vectors) {
    std::ofstream f(path, std::ios::binary);
    std::string h = "{'descr': '<f4', 'fortran_order': False, 'shape': ("+
                    std::to_string(vectors.size())+", 512), }";
    h.append((64-(10+h.size()+1)%64)%64, ' '); h+='\n';
    f.write("\x93NUMPY\x01\x00",8);
    unsigned char len[2] = {static_cast<unsigned char>(h.size()&255),
                           static_cast<unsigned char>(h.size()>>8)};
    f.write(reinterpret_cast<char*>(len),2); f.write(h.data(),h.size());
    for (const auto& v : vectors)
        f.write(reinterpret_cast<const char*>(v.data()), sizeof(v));
    require(bool(f), "Cannot save embeddings");
}

class Model {
public:
    rknn_context ctx = 0;
    std::vector<rknn_tensor_attr> attrs;
    Model(const std::string& path, int side) {
        try {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            require(bool(f), "Cannot open model: "+path);
            auto size = f.tellg();
            require(size>0 && size<2147483647, "Invalid model size");
            std::vector<char> data(static_cast<size_t>(size));
            f.seekg(0); f.read(data.data(),data.size());
            require(bool(f), "Cannot read model");
            check(rknn_init(&ctx,data.data(),data.size(),0,nullptr),"rknn_init");
            check(rknn_set_core_mask(ctx,RKNN_NPU_CORE_0_1),"core mask");
            rknn_input_output_num io{};
            check(rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io)),"IO query");
            require(io.n_input==1 && io.n_output>0 && io.n_output<=2,
                    "Unexpected model input/output count");
            rknn_tensor_attr input{};
            check(rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&input,sizeof(input)),
                  "input query");
            bool nchw = input.fmt==RKNN_TENSOR_NCHW && input.n_dims==4 &&
                input.dims[0]==1 && input.dims[1]==3 &&
                input.dims[2]==unsigned(side) && input.dims[3]==unsigned(side);
            bool nhwc = input.fmt==RKNN_TENSOR_NHWC && input.n_dims==4 &&
                input.dims[0]==1 && input.dims[1]==unsigned(side) &&
                input.dims[2]==unsigned(side) && input.dims[3]==3;
            require(nchw || nhwc,"Unexpected model input shape/layout");
            // MBF was converted with dynamic_input; select its shape before IO.
            if (side==112) {
                check(rknn_set_input_shapes(ctx,1,&input),"set MBF input shape");
            }
            attrs.resize(io.n_output);
            for (unsigned i=0;i<io.n_output;++i) {
                attrs[i].index=i;
                check(rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&attrs[i],
                                sizeof(attrs[i])),"output query");
                std::cout<<path<<" output "<<i<<":";
                for (unsigned j=0;j<attrs[i].n_dims;++j)
                    std::cout<<" "<<attrs[i].dims[j];
                std::cout<<std::endl;
            }
        } catch (...) { if(ctx) rknn_destroy(ctx); ctx=0; throw; }
    }
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    ~Model() { if(ctx) rknn_destroy(ctx); }
    // The RKNN models embed mean/std; supply RGB UINT8, pass_through=0.
    std::vector<std::vector<float>> run(const cv::Mat& rgb) {
        require(rgb.type()==CV_8UC3 && rgb.isContinuous(),"Invalid RGB input");
        rknn_input input{};
        input.type=RKNN_TENSOR_UINT8; input.fmt=RKNN_TENSOR_NHWC;
        input.size=rgb.total()*3; input.buf=rgb.data;
        check(rknn_inputs_set(ctx,1,&input),"inputs_set");
        check(rknn_run(ctx,nullptr),"run");
        std::vector<rknn_output> out(attrs.size());
        for(unsigned i=0;i<out.size();++i) {
            out[i].index=i; out[i].want_float=1;
        }
        check(rknn_outputs_get(ctx,out.size(),out.data(),nullptr),"outputs_get");
        std::vector<std::vector<float>> result;
        try {
            for(unsigned i=0;i<out.size();++i) {
                require(out[i].buf && out[i].size>=attrs[i].n_elems*sizeof(float),
                        "Invalid model output buffer");
                const float* p=static_cast<float*>(out[i].buf);
                result.emplace_back(p,p+attrs[i].n_elems);
            }
        } catch (...) { rknn_outputs_release(ctx,out.size(),out.data()); throw; }
        check(rknn_outputs_release(ctx,out.size(),out.data()),"outputs_release");
        return result;
    }
};
struct Face { cv::Rect2f box; float confidence; };
class Detector {
    Model model;
    int bi=-1, si=-1;
    bool channel_first=true;
public:
    explicit Detector(const std::string& path):model(path,640) {
        require(model.attrs.size()==2,"YOLO requires split boxes/scores");
        for(unsigned i=0;i<model.attrs.size();++i) {
            auto a=model.attrs[i];
            if(a.n_dims!=3 || a.dims[0]!=1) continue;
            if(a.n_elems==4*8400 &&
                ((a.dims[1]==4 && a.dims[2]==8400) ||
                 (a.dims[1]==8400 && a.dims[2]==4))) {
                bi=i; channel_first=a.dims[1]==4;
            } else if(a.n_elems==8400 &&
                ((a.dims[1]==1 && a.dims[2]==8400) ||
                 (a.dims[1]==8400 && a.dims[2]==1))) si=i;
        }
        require(bi>=0 && si>=0,"Unsupported split YOLO output layout");
    }
    std::vector<Face> detect(const cv::Mat& frame,float conf,float nms) {
        cv::Mat rgb; cv::resize(frame,rgb,{640,640});
        cv::cvtColor(rgb,rgb,cv::COLOR_BGR2RGB);
        auto outputs=model.run(rgb);
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        for(int i=0;i<8400;++i) {
            float s=outputs[si][i];
            if(!std::isfinite(s) || s<conf) continue;
            auto at=[&](int c){return outputs[bi][channel_first?c*8400+i:i*4+c];};
            float cx=at(0),cy=at(1),w=at(2),h=at(3);
            if(!std::isfinite(cx)||!std::isfinite(cy)||!std::isfinite(w)||
               !std::isfinite(h)||w<=0||h<=0) continue;
            double x1=std::clamp((cx-w/2)*frame.cols/640.0,0.0,double(frame.cols));
            double y1=std::clamp((cy-h/2)*frame.rows/640.0,0.0,double(frame.rows));
            double x2=std::clamp((cx+w/2)*frame.cols/640.0,0.0,double(frame.cols));
            double y2=std::clamp((cy+h/2)*frame.rows/640.0,0.0,double(frame.rows));
            if(x2<=x1 || y2<=y1) continue;
            boxes.emplace_back(x1,y1,x2-x1,y2-y1); scores.push_back(s);
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes,scores,conf,nms,keep);
        std::vector<Face> faces;
        for(int i:keep) faces.push_back({cv::Rect2f(boxes[i]),scores[i]});
        return faces;
    }
};
class Embedder {
    Model model;
public:
    explicit Embedder(const std::string& path):model(path,112) {
        require(model.attrs.size()==1 && model.attrs[0].n_elems==512,
                "MBF must return one 512-dimensional vector");
    }
    Vector embed(const cv::Mat& crop) {
        cv::Mat rgb; cv::resize(crop,rgb,{112,112});
        cv::cvtColor(rgb,rgb,cv::COLOR_BGR2RGB);
        auto outputs=model.run(rgb);
        Vector v; std::copy(outputs[0].begin(),outputs[0].end(),v.begin());
        return normalize(v);
    }
};
// Single-memory-plane NV12 on the Rockchip multiplanar capture interface.
class Camera {
    int fd=-1,width=0,height=0,stride=0;
    bool streaming=false;
    struct Buffer { void* data=MAP_FAILED; size_t size=0; };
    std::vector<Buffer> buffers;
    int call(unsigned long op,void* arg) {
        int r; do {r=ioctl(fd,op,arg);} while(r<0 && errno==EINTR); return r;
    }
public:
    Camera()=default;
    Camera(const Camera&)=delete;
    ~Camera(){close();}
    void close() {
        if(fd<0) return;
        if(streaming) {v4l2_buf_type t=V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                      call(VIDIOC_STREAMOFF,&t);}
        for(auto b:buffers) if(b.data!=MAP_FAILED) munmap(b.data,b.size);
        ::close(fd); fd=-1; streaming=false; buffers.clear();
    }
    void open(const std::string& device,int w,int h) {
        fd=::open(device.c_str(),O_RDWR|O_NONBLOCK);
        require(fd>=0,"Cannot open camera: "+device);
        v4l2_format fmt{}; fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        fmt.fmt.pix_mp.width=w; fmt.fmt.pix_mp.height=h;
        fmt.fmt.pix_mp.pixelformat=V4L2_PIX_FMT_NV12;
        fmt.fmt.pix_mp.field=V4L2_FIELD_NONE; fmt.fmt.pix_mp.num_planes=1;
        require(call(VIDIOC_S_FMT,&fmt)==0,"Camera format failed");
        auto p=fmt.fmt.pix_mp;
        require(p.pixelformat==V4L2_PIX_FMT_NV12 && p.num_planes==1 &&
                p.width%2==0 && p.height%2==0,"Need single-plane NV12");
        width=p.width; height=p.height; stride=p.plane_fmt[0].bytesperline;
        require(stride>=width,"Invalid camera stride");
        v4l2_requestbuffers req{}; req.count=3;
        req.type=fmt.type; req.memory=V4L2_MEMORY_MMAP;
        require(call(VIDIOC_REQBUFS,&req)==0 && req.count>0,"Camera buffers failed");
        buffers.resize(req.count);
        for(unsigned i=0;i<req.count;++i) {
            v4l2_plane plane{}; v4l2_buffer b{};
            b.type=fmt.type; b.memory=V4L2_MEMORY_MMAP;
            b.index=i; b.length=1; b.m.planes=&plane;
            require(call(VIDIOC_QUERYBUF,&b)==0,"Camera query buffer failed");
            buffers[i].size=plane.length;
            buffers[i].data=mmap(nullptr,plane.length,PROT_READ|PROT_WRITE,
                                 MAP_SHARED,fd,plane.m.mem_offset);
            require(buffers[i].data!=MAP_FAILED,"Camera mmap failed");
            require(call(VIDIOC_QBUF,&b)==0,"Camera queue failed");
        }
        v4l2_buf_type t=V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        require(call(VIDIOC_STREAMON,&t)==0,"Camera stream start failed");
        streaming=true;
        std::cout<<"Camera NV12 "<<width<<"x"<<height<<" stride="<<stride<<std::endl;
    }
    cv::Mat read() {
        pollfd p{fd,POLLIN,0}; int r;
        do {r=poll(&p,1,2000);} while(r<0 && errno==EINTR);
        require(r>0 && !(p.revents&(POLLERR|POLLHUP|POLLNVAL)),"Camera timeout/error");
        v4l2_plane plane{}; v4l2_buffer b{};
        b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        b.memory=V4L2_MEMORY_MMAP; b.length=1; b.m.planes=&plane;
        require(call(VIDIOC_DQBUF,&b)==0,"Camera dequeue failed");
        cv::Mat frame;
        try {
            require(b.index<buffers.size() && !(b.flags&V4L2_BUF_FLAG_ERROR),
                    "Invalid camera frame");
            size_t needed=size_t(stride)*height*3/2;
            require(plane.data_offset<=buffers[b.index].size &&
                    needed<=buffers[b.index].size-plane.data_offset &&
                    plane.bytesused>=needed+plane.data_offset,"Short NV12 frame");
            auto data=static_cast<unsigned char*>(buffers[b.index].data)+plane.data_offset;
            cv::Mat y(height,width,CV_8UC1,data,stride);
            cv::Mat uv(height/2,width/2,CV_8UC2,data+size_t(stride)*height,stride);
            cv::cvtColorTwoPlane(y,uv,frame,cv::COLOR_YUV2BGR_NV12);
        } catch(...) {call(VIDIOC_QBUF,&b);throw;}
        require(call(VIDIOC_QBUF,&b)==0,"Camera requeue failed");
        return frame;
    }
};
struct Options {
    std::string yolo="/home/industio/yolo_test/yolo11n_face_split_int8.rknn";
    std::string mbf="/home/industio/yolo_test/w600k_mbf_rk3576_fp16.rknn";
    std::string reference="/home/industio/yolo_test/embedding_reference/owner.npy";
    std::string camera="/dev/video42",output,image;
    double threshold=0,duration=3,sample_fps=5,margin=.2;
    float conf=.25f,nms=.45f;
    int max_faces=50,min_face=40,width=1920,height=1080;
    bool threshold_set=false,check_only=false;
};
static Options options(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--help") {
            std::cout<<"face_recognition --threshold VALUE --output NEW_DIR [options]\n"
              "--yolo PATH --mbf PATH --reference PATH --camera /dev/video42\n"
              "--duration 3 --sample-fps 5 --max-faces 50 --min-face-px 40\n"
              "--margin 0.2 --conf 0.25 --nms-iou 0.45 --width 1920 --height 1080\n"
              "--check-only (load models/reference only)\n"
              "--image PATH (single-image inference, no camera)\n";
            std::exit(0);
        }
        if(key=="--check-only") {o.check_only=true;continue;}
        require(i+1<argc,"Missing value: "+key); std::string v=argv[++i];
        if(key=="--yolo") o.yolo=v;
        else if(key=="--mbf") o.mbf=v;
        else if(key=="--reference") o.reference=v;
        else if(key=="--camera") o.camera=v;
        else if(key=="--output") o.output=v;
        else if(key=="--image") o.image=v;
        else if(key=="--threshold") {o.threshold=std::stod(v);o.threshold_set=true;}
        else if(key=="--duration") o.duration=std::stod(v);
        else if(key=="--sample-fps") o.sample_fps=std::stod(v);
        else if(key=="--margin") o.margin=std::stod(v);
        else if(key=="--conf") o.conf=std::stof(v);
        else if(key=="--nms-iou") o.nms=std::stof(v);
        else if(key=="--max-faces") o.max_faces=std::stoi(v);
        else if(key=="--min-face-px") o.min_face=std::stoi(v);
        else if(key=="--width") o.width=std::stoi(v);
        else if(key=="--height") o.height=std::stoi(v);
        else throw std::runtime_error("Unknown option: "+key);
    }
    require(std::isfinite(o.duration)&&o.duration>0 && o.duration<=3600 &&
            std::isfinite(o.sample_fps)&&o.sample_fps>0 && o.sample_fps<=1000 &&
            std::isfinite(o.margin)&&o.margin>=0 && o.margin<=10 &&
            std::isfinite(o.conf)&&o.conf>=0&&o.conf<=1 &&
            std::isfinite(o.nms)&&o.nms>=0&&o.nms<=1 &&
            o.max_faces>0&&o.max_faces<=10000&&o.min_face>0 &&
            o.width>0&&o.height>0,"Invalid capture settings");
    require(o.check_only || (o.threshold_set&&std::isfinite(o.threshold)&&
            o.threshold>=-1&&o.threshold<=1&&!o.output.empty()),
            "Provide calibrated --threshold and --output");
    return o;
}
struct Crop {cv::Mat image; cv::Rect2f box; float confidence; int frame; double seconds;};
int main(int argc,char** argv) {
    try {
        Options o=options(argc,argv);
        Vector reference=load_reference(o.reference);
        auto load_start=Clock::now();
        auto detector=std::make_unique<Detector>(o.yolo);
        Embedder mbf(o.mbf);
        double model_load_ms=ms(load_start);
        if(o.check_only) {
            std::cout<<"CHECK_OK: reference and both model contracts validated; "
                       "no inference or camera opened\n"; return 0;
        }
        fs::path output=o.output;
        require(!fs::exists(output),"Output already exists: "+o.output);
        fs::create_directories(output);
        std::vector<Crop> cache;
        int sampled=0,skipped=0;
        int target=o.image.empty()?int(std::ceil(o.duration*o.sample_fps)):1;
        auto collect=[&](const cv::Mat& frame,double seconds) {
            if(sampled==0) require(cv::imwrite((output/"preview.jpg").string(),frame),
                                   "Cannot save preview");
            auto faces=detector->detect(frame,o.conf,o.nms); ++sampled;
            for(const auto& face:faces) {
                auto b=face.box;
                if(std::min(b.width,b.height)<o.min_face) continue;
                int x1=int(std::max(0.0,double(b.x-b.width*o.margin)));
                int y1=int(std::max(0.0,double(b.y-b.height*o.margin)));
                int x2=int(std::min(double(frame.cols),std::ceil(b.x+b.width*(1+o.margin))));
                int y2=int(std::min(double(frame.rows),std::ceil(b.y+b.height*(1+o.margin))));
                if(x2<=x1||y2<=y1) continue;
                cache.push_back({frame(cv::Rect(x1,y1,x2-x1,y2-y1)).clone(),
                                 b,face.confidence,sampled,seconds});
                if(int(cache.size())>=o.max_faces) break;
            }
            std::cout<<"Sample "<<sampled<<" detected="<<faces.size()
                     <<" cached="<<cache.size()<<std::endl;
        };
        double capture_ms=0;
        if(!o.image.empty()) {
            auto start=Clock::now(); cv::Mat frame=cv::imread(o.image);
            require(!frame.empty(),"Cannot read image"); collect(frame,0);
            capture_ms=ms(start);
        } else {
            Camera camera; camera.open(o.camera,o.width,o.height);
            auto warmup=Clock::now();
            while(ms(warmup)<2000) camera.read();
            auto start=Clock::now(); double next_due=0;
            while(sampled<target && int(cache.size())<o.max_faces &&
                  ms(start)<o.duration*1000) {
                cv::Mat frame=camera.read(); double elapsed=ms(start)/1000;
                if(elapsed>=o.duration) break;
                if(elapsed<next_due) continue;
                collect(frame,elapsed); next_due+=1/o.sample_fps;
                double now=ms(start)/1000;
                if(next_due<now) {
                    int missed=int(std::ceil((now-next_due)*o.sample_fps));
                    skipped+=missed; next_due+=missed/o.sample_fps;
                }
            }
            capture_ms=ms(start); camera.close();
        }
        detector.reset(); // Stop YOLO before the cached-face recognition pass.
        cv::FileStorage report((output/"results.json").string(),
                               cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
        require(report.isOpened(),"Cannot create report");
        report<<"reference"<<o.reference<<"threshold"<<o.threshold;
        report<<"target_frames"<<target<<"sampled_frames"<<sampled;
        report<<"skipped_sample_slots"<<skipped<<"cached_faces"<<int(cache.size());
        report<<"capture_seconds"<<capture_ms/1000<<"model_load_ms"<<model_load_ms;
        report<<"note"<<"Per-crop matches only; no tracking, alignment, liveness or session-level identity decision.";
        report<<"timing_note"<<"Recognition phase includes crop file writes; excludes model loads and final report/NPY writes.";
        report<<"results"<<"[";
        std::vector<Vector> vectors;
        auto recognition_start=Clock::now();
        for(size_t i=0;i<cache.size();++i) {
            auto start=Clock::now(); Vector v=mbf.embed(cache[i].image);
            double score=0; for(size_t j=0;j<v.size();++j) score+=double(v[j])*reference[j];
            score=std::clamp(score,-1.0,1.0); double elapsed=ms(start);
            std::string name=cv::format("face_%04d.png",int(i));
            require(cv::imwrite((output/name).string(),cache[i].image),"Crop write failed");
            vectors.push_back(v); auto b=cache[i].box;
            report<<"{"<<"crop"<<name<<"sampled_frame"<<cache[i].frame;
            report<<"capture_seconds"<<cache[i].seconds;
            report<<"box_xyxy"<<"["<<b.x<<b.y<<b.x+b.width<<b.y+b.height<<"]";
            report<<"detection_confidence"<<cache[i].confidence<<"cosine"<<score;
            report<<"match"<<int(score>=o.threshold)<<"embedding_compare_ms"<<elapsed<<"}";
            std::cout<<name<<" cosine="<<score<<" match="<<(score>=o.threshold)
                     <<" embedding_compare_ms="<<elapsed<<std::endl;
        }
        double recognition_ms=ms(recognition_start);
        report<<"]"<<"recognition_phase_ms"<<recognition_ms;
        report<<"recognition_performed"<<int(!cache.empty()); report.release();
        save_embeddings(output/"embeddings.npy",vectors);
        std::cout<<"Cached faces="<<cache.size()<<" recognition_phase_ms="
                 <<recognition_ms<<" output="<<output<<std::endl;
        if(cache.empty()) std::cout<<"No faces: recognition latency is not a benchmark.\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1;}
}
