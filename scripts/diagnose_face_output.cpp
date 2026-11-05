#include <cstdio>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
int main(int argc,char**argv){
 if(argc!=3)return 2;
 FILE*f=fopen(argv[1],"rb");if(!f)return 3;
 fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);
 std::vector<unsigned char> m(n);if(fread(m.data(),1,n,f)!=(size_t)n)return 4;fclose(f);
 rknn_context ctx=0;int ret=rknn_init(&ctx,m.data(),n,0,nullptr);printf("init=%d\n",ret);if(ret)return 5;
 rknn_input_output_num io{};ret=rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io));if(ret)return 6;
 printf("inputs=%u outputs=%u\n",io.n_input,io.n_output);
 rknn_tensor_attr a{};
 for(unsigned j=0;j<io.n_output;j++){a={};a.index=j;ret=rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&a,sizeof(a));if(ret)return 7;
 printf("output%u type=%d quant=%d scale=%.9g zp=%d elems=%u dims=",j,a.type,a.qnt_type,a.scale,a.zp,a.n_elems);for(unsigned d=0;d<a.n_dims;d++)printf("%u,",a.dims[d]);puts("");}
 if(io.n_output!=1||a.n_elems!=5*8400)return 8;
 cv::Mat bgr=cv::imread(argv[2]),rgb;if(bgr.empty())return 9;
 cv::resize(bgr,bgr,cv::Size(640,640));cv::cvtColor(bgr,rgb,cv::COLOR_BGR2RGB);
 rknn_input in{};in.type=RKNN_TENSOR_UINT8;in.fmt=RKNN_TENSOR_NHWC;in.size=640*640*3;in.buf=rgb.data;
 ret=rknn_inputs_set(ctx,1,&in);printf("inputs_set=%d\n",ret);if(ret)return 10;
 ret=rknn_run(ctx,nullptr);printf("run=%d\n",ret);if(ret)return 11;
 for(int fl=0;fl<=1;fl++){
 rknn_output out{};out.want_float=fl;ret=rknn_outputs_get(ctx,1,&out,nullptr);printf("get float=%d ret=%d bytes=%u\n",fl,ret,out.size);if(ret)return 12;
 for(int c=0;c<5;c++){double lo=1e30,hi=-1e30;int above=0;for(int i=0;i<8400;i++){int k=c*8400+i;double v;
 if(fl)v=((float*)out.buf)[k];else if(a.type==RKNN_TENSOR_INT8)v=((int8_t*)out.buf)[k];else if(a.type==RKNN_TENSOR_UINT8)v=((uint8_t*)out.buf)[k];else{break;}
 lo=std::min(lo,v);hi=std::max(hi,v);if(v>=0.25)above++;}
 printf("float=%d channel=%d min=%.9g max=%.9g above025=%d\n",fl,c,lo,hi,above);}
 rknn_outputs_release(ctx,1,&out);
 }
 rknn_destroy(ctx);return 0;
}
