/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "./centernet_imp.hpp"

#include <fstream>
#include <iostream>
#include <queue>
#include <sys/stat.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

DEF_ENV_PARAM(ENABLE_CN_DEBUG, "0");
DEF_ENV_PARAM(XLNX_CNET_TOPK, "100");
DEF_ENV_PARAM(XLNX_CNET_WHSCALE, "128");
DEF_ENV_PARAM(XLNX_CNET_IMPORT_POST, "0");
DEF_ENV_PARAM_2(XLNX_CNET_THRESH,    "0.01", float );

using namespace std;
using namespace cv;

namespace vitis {
namespace ai {

inline float sigmoid(float x) {
  return 1.0/(1.0+exp(-x) );
}

std::string type2str(int type) {
  std::string r;
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }
  r += "C";
  r += (chans+'0');
  return r;
}

CenterNetImp::CenterNetImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<CenterNet>(model_name, need_preprocess),
      input_tensors_ (configurable_dpu_task_->getInputTensor()[0]),
      output_tensors_(configurable_dpu_task_->getOutputTensor()[0])
{

// in: HWC scale :  512 512  3 32
// out :  HWC : scale  128  128  80 0.25      # hm   # heat==score ? max100 and its position and classes
// out :  HWC : scale  128  128  2 0.0078125  # reg  # below 2 for the boxes.
// out :  HWC : scale  128  128  2 1          # wh

  if (ENV_PARAM(ENABLE_CN_DEBUG) == 1) {
    {
      const auto& layer_data = input_tensors_[0];
      int sWidth = layer_data.width;
      int sHeight= layer_data.height;
      auto channels = layer_data.channel;
      float scale =  tensor_scale(layer_data);
      std::cout <<"in: HWC scale :  " <<sHeight << " " << sWidth << "  " << channels << " " << scale <<"\n";  
    }

    for (int i=0; i<3; i++) {
      const auto& layer_data = output_tensors_[i];
      int sWidth = layer_data.width;
      int sHeight= layer_data.height;
      auto channels = layer_data.channel;
      float scale =  tensor_scale(layer_data);
      std::cout <<"out :  HWC : scale  " << sHeight << "  " << sWidth << "  " << channels << " " << scale <<"\n";
    }
  }

  iwidth = input_tensors_[0].width;
  iheight = input_tensors_[0].height;
  for(int i=0; i<3; i++) {
     if((int)output_tensors_[i].channel == num_classes ) {
        idx_hm = i;  
        continue;
     }
     else if( output_tensors_[i].name.find("reg") != std::string::npos) {
        idx_reg = i;  
        continue;
     }
  }
  for(int i=0; i<3; i++) { 
     if (i != idx_hm && i != idx_reg ) {
        idx_wh = i; 
        break;
     }
  }
  topk = ENV_PARAM(XLNX_CNET_TOPK);
  batch_size = get_input_batch();
#if TESTF
  oscale_hm = 1.0;
  oscale_wh = 1.0 * ENV_PARAM(XLNX_CNET_WHSCALE);
  oscale_reg = 1.0;
#else
  oscale_hm  = tensor_scale(  output_tensors_[idx_hm]);
  oscale_wh  = tensor_scale(  output_tensors_[idx_wh]) * ENV_PARAM(XLNX_CNET_WHSCALE);
  oscale_reg = tensor_scale(  output_tensors_[idx_reg]);
#endif

  score_threshold = ENV_PARAM(XLNX_CNET_THRESH);
  score_threshold_f = (-log(1.0/score_threshold -1.0 ))/oscale_hm;

  oData_hm.resize(batch_size);
  oData_wh.resize(batch_size);
  oData_reg.resize(batch_size);
  img_h.resize(batch_size);
  img_w.resize(batch_size);
  img_maxhw.resize(batch_size);
  for(int i=0; i<batch_size; i++) {
#if TESTF
     oData_hm[i] = new float[128*128*80];
     oData_wh[i] = new float[128*128*2];
     oData_reg[i] = new float[128*128*2];
#else
     oData_hm[i]  = (int8_t*)output_tensors_[idx_hm].get_data(i);
     oData_wh[i]  = (int8_t*)output_tensors_[idx_wh].get_data(i);
     oData_reg[i] = (int8_t*)output_tensors_[idx_reg].get_data(i);
#endif
  }

  owidth  =  output_tensors_[0].width;
  oheight =  output_tensors_[0].height;

  // std::cout <<" idx hm wh reg : " << idx_hm << " " << idx_wh << " " << idx_reg << "\n";
}

CenterNetImp::~CenterNetImp() {}

#if 0
struct CenterNetResult{
  struct Det_Res{ float score; float pos[4]; }
  std::vector<std::vector<Det_Res>> vres;
}
#endif

std::vector<CenterNetResult> CenterNetImp::centernet_post_process() {
  auto ret = std::vector<vitis::ai::CenterNetResult>{};
  ret.reserve(real_batch_size);
  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(centernet_post_process(i));
  }
  return ret;
}


int getfilelen(const std::string& file)
{
  struct stat statbuf;
  if(stat(file.c_str(), &statbuf)!=0){
    std::cerr << " bad file stat " << file << std::endl;
    exit(-1);
  }
  return statbuf.st_size;
}

template<typename T>
void myreadfile(T* conf, int size1, std::string filename){
  ifstream Tin;
  Tin.open(filename, ios_base::in|ios_base::binary);
  if(!Tin)  {      cout<<"Can't open the file!";      return ;  }
  Tin.read( (char*)conf, size1*sizeof(T));
}

CenterNetResult CenterNetImp::centernet_post_process(int idx) {

#if TESTF
     std::string filen="./cn_hmf.bin";
     myreadfile( oData_hm[idx] , getfilelen(filen)/4,  filen);
     filen = "./cn_whf.bin";
     myreadfile( oData_wh[idx] , getfilelen(filen)/4,  filen);
     filen = "./cn_regf.bin";
     myreadfile( oData_reg[idx] , getfilelen(filen)/4,  filen);
#else
  if (ENV_PARAM(XLNX_CNET_IMPORT_POST)) {
     std::string filen="./cn_hm.bin";
     myreadfile( oData_hm[idx] , getfilelen(filen),  filen);
     filen = "./cn_wh.bin";
     myreadfile( oData_wh[idx] , getfilelen(filen),  filen);
     filen = "./cn_reg.bin";
     myreadfile( oData_reg[idx] , getfilelen(filen),  filen);
  }
#endif

  CenterNetResult ret{int(input_tensors_[0].width), int(input_tensors_[0].height),  
                      std::vector<std::vector<CenterNetResult::Det_Res>>(num_classes) };

#if TESTF
  struct cmp1 {
    bool operator ()(const std::pair<int, float>& a, const std::pair<int, float>& b ) {
      return std::get<1>(a) >  std::get<1>(b);
    }
  };
  priority_queue<std::pair<int,float>, vector<std::pair<int, float>>,cmp1> minHeap;
#else
  struct cmp1 {
    bool operator ()(const std::pair<int, int8_t>& a, const std::pair<int, int8_t>& b ) {
      return std::get<1>(a) >  std::get<1>(b);
    }
  };
  priority_queue<std::pair<int,int8_t>, vector<std::pair<int, int8_t>>,cmp1> minHeap;
#endif

  // for(int i=0; i<128; i++) {
  //   for(int j=0; j<128; j++) {
  //     std::cout <<i << " " << j << "    " << oData_wh[idx][ (i*128+j)*2+0 ]* oscale_wh << " " ;
  //     std::cout <<oData_wh[idx][ (i*128+j)*2+1 ]* oscale_wh << "\n" ;
  //   }
  // }

  const int kernel_size = 3;
#if TESTF
  float tmp_v, topv=0.0;
#else
  int8_t tmp_v, topv=0;
#endif
  int loop = 0, pos = 0;

  bool foundbig = false;
  int owidth_num_classes = owidth*num_classes;
  __TIC__(maxpool)
  for (auto i = 0; i < oheight; ++i) {
    for (auto j = 0; j < owidth; ++j) {
      for (auto k = 0; k < num_classes; ++k) {
        foundbig = false;
        pos = i*owidth_num_classes + j*num_classes + k;
        tmp_v = oData_hm[idx][pos];

#if 1
        if ( (float)tmp_v < score_threshold_f ) {
           continue;
        }
#endif

#if 1
        // if (loop > topk && tmp_v <= std::get<1>(minHeap.top())) {
        if (loop > topk && tmp_v <= topv ) {
           continue;
        } 
#endif
        for (int di = 0; di < kernel_size; di++) {
          for (int dj = 0; dj < kernel_size; dj++) {
            auto input_h_idx = ((i - 1) + di);
            auto input_w_idx = ((j - 1) + dj);
            if (    input_w_idx < 0
                 || input_h_idx < 0
                 || input_h_idx >= oheight 
                 || input_w_idx >= owidth 
                 || (input_h_idx == i && input_w_idx == j) ) {
              continue;
            }
            if ( oData_hm[idx][input_h_idx * owidth_num_classes + input_w_idx *num_classes + k ] > tmp_v) {
               foundbig = true;
               goto breakfound;
            }
          }
        }
    breakfound:
        if (!foundbig) {

          // std::cout <<"maxpool : " << i << " " << j << " " << k << "  " << sigmoid(tmp_v*oscale_hm)<< "\n";

          if (loop <= topk) {
             loop++;
             minHeap.push(std::make_pair(pos, tmp_v) );
             /// continue;
#if 0
          }
          if (tmp_v<= std::get<1>(minHeap.top())) {
             continue;
#endif
          } else { // (tmp_v> std::get<1>(minHeap.top()))
             minHeap.pop();
             minHeap.push(std::make_pair(pos, tmp_v) );
          }
          topv = std::get<1>(minHeap.top());
        }
      }
    }
  }
  __TOC__(maxpool)

  // std::cout << "loop : " << loop <<"\n"; // 210000

  cv::Mat m1 = get_affine_transform(img_w[idx]*0.5, img_h[idx]*0.5, img_maxhw[idx], owidth, oheight, true);

  int key = 0;
  int xsi = 0, ysi = 0, cls = 0;  
  float score = 0.0, xs = 0.0, ys = 0.0, tmp0, tmp1; 
  std::array<float, 4> coord;
  __TIC__(topk)
  while(!minHeap.empty()){
    key = std::get<0>(minHeap.top());
    score = sigmoid(std::get<1>(minHeap.top())*oscale_hm);
    minHeap.pop();

    ysi = key/(owidth*num_classes);
    xsi = (key % (owidth*num_classes) )/num_classes;
    cls = (key % (owidth*num_classes) ) % num_classes;
    pos = ysi*owidth*2+xsi*2;

  // std::cout <<" yxc : " << ysi << " " << xsi << " " << cls << "  " << score << "\n";
    xs = xsi + (oData_reg[idx][pos+0 ]*oscale_reg);  // 128x128x2 
    ys = ysi + (oData_reg[idx][pos+1 ]*oscale_reg);  // 128x128x2 

    //  # print("c s h w num_classes ", c, s, h, w, num_classes )#   [array([176., 115.], )] [352.0] 128 128 80
    //  dets[0, :, :2] = transform_preds( dets[0, :, 0:2], c[0], s[0], (w, h)) # [ 100 2 ]
    //  dets[0, :, 2:4] = transform_preds( dets[0, :, 2:4], c[0], s[0], (w, h))

    tmp0 = xs - oData_wh[idx][pos+0 ]*oscale_wh/2.0 ;
    tmp1 = ys - oData_wh[idx][pos+1 ]*oscale_wh/2.0 ;
    coord[0] = tmp0*m1.ptr<double>(0)[0] +  tmp1*m1.ptr<double>(0)[1] + m1.ptr<double>(0)[2];
    coord[1] = tmp0*m1.ptr<double>(1)[0] +  tmp1*m1.ptr<double>(1)[1] + m1.ptr<double>(1)[2];
    tmp0 = xs + oData_wh[idx][pos+0 ]*oscale_wh/2.0 ;
    tmp1 = ys + oData_wh[idx][pos+1 ]*oscale_wh/2.0 ;
    coord[2] = tmp0*m1.ptr<double>(0)[0] +  tmp1*m1.ptr<double>(0)[1] + m1.ptr<double>(0)[2];
    coord[3] = tmp0*m1.ptr<double>(1)[0] +  tmp1*m1.ptr<double>(1)[1] + m1.ptr<double>(1)[2];

    ret.vres[cls].emplace_back( CenterNetResult::Det_Res(score, coord));
  }
  __TOC__(topk)
  // for(int i=0 ; i<num_classes; i++) { std::cout <<"ret : " << i << " " << ret.vres[i].size() << "\n"; }
  return ret;
}

cv::Mat CenterNetImp::get_affine_transform(float center_x, float center_y, float scale, int o_w, int o_h, bool inv) {
  float dst_w = o_w;
  float dst_h = o_h; 
  float src_dir1 = scale* (-0.5);
  float dst_dir1 = dst_w * (-0.5);

// std::cout <<" get_affine_t : " << center_x << " " << center_y << " " << scale << " " << o_w << " " << o_h << " " << inv << "\n";  //  320 213.5 640 512 512 0

  auto get_3rd_point = [=](Point2f& a, Point2f& b) -> Point2f {  
    Point2f direct = a-b;
    return b+Point2f( -direct.y, direct.x );
  };
  Point2f src[3] = {{center_x, center_y}, 
                    { center_x , float(center_y + src_dir1) }, 
                    get_3rd_point(src[0],  src[1])  };
  Point2f dst[3] = {{float(dst_w*0.5), float(dst_h*0.5)}, 
                    {float(dst_w*0.5) , float( dst_h*0.5 + dst_dir1 ) }, 
                    get_3rd_point(dst[0], dst[1]) };

  //for(int i=0; i<3; i++) std::cout << "src " << src[i].x << " " << src[i].y << " \n";
  //for(int i=0; i<3; i++) std::cout << "dst " << dst[i].x << " " << dst[i].y << " \n";
  cv::Mat ret = inv ? cv::getAffineTransform( dst, src ) : cv::getAffineTransform( src, dst );

#if 0
  for(int i=0; i<2; i++) {
    for(int j=0; j<3; j++) {
      std::cout << ret.ptr<double>(i)[j] << " "; 
    }
  } std::cout <<" ----- \n";
#endif
  // std::cout <<"ret hw : " << ret.rows << " " << ret.cols << "\n";  // 2 3
  // std::cout << "type :" <<  type2str( ret.type() ) <<"\n";   // 64FC1

  return ret;
}

cv::Mat CenterNetImp::preprocess(const cv::Mat img, int idx ) {
  img_h[idx] = img.rows;
  img_w[idx] = img.cols;
  img_maxhw[idx] = std::max(img_h[idx], img_w[idx] );
  __TIC__(pre)
  cv::Mat warp_trans = get_affine_transform(img.cols/2.0, img.rows/2.0, std::max(img.cols, img.rows) , iwidth, iheight, false);
  cv::Mat warp_dst(  iheight, iwidth, CV_8UC1 , cvScalar(0) );
  cv::warpAffine( img, warp_dst, warp_trans, warp_dst.size() ); 
  __TOC__(pre)
 
#if 0  
  //  std::cout << "type :" <<  type2str( warp_dst.type() ) <<"\n";   // 8UC3
  for (int i=0; i<iheight; i++) {
    for (int j=0; j<iwidth; j++) {
      cv::Vec3b x= warp_dst.at<Vec3b>(i,j);
      std::cout <<i << " " << j << "   " << (int)x[0] << " " << (int)x[1] << " " << (int)x[2] <<"\n";
    }
  }
#endif
  return warp_dst;
}

CenterNetResult CenterNetImp::run( const cv::Mat &input_img) {
  __TIC__(CenterNet_total)

  cv::Mat img;
  __TIC__(centernet_pre)
  img = preprocess(input_img, 0);
  __TOC__(centernet_pre)

  __TIC__(CenterNet_setimg)
  real_batch_size = 1;
  configurable_dpu_task_->setInputImageRGB(img);
  __TOC__(CenterNet_setimg)

  __TIC__(CenterNet_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(CenterNet_dpu)

  __TIC__(CenterNet_post)
  auto results = centernet_post_process();
  __TOC__(CenterNet_post)

  __TOC__(CenterNet_total)
  return results[0];
}

std::vector<CenterNetResult> CenterNetImp::run( const std::vector<cv::Mat> &input_img) {
  __TIC__(CenterNet_total)
  __TIC__(centernet_pre)
  real_batch_size = std::min(int(input_img.size()), int(batch_size));
  std::vector<cv::Mat> vimg(real_batch_size);
  for (auto i = 0; i < real_batch_size; i++) {
    vimg[i] = preprocess(input_img[i], i );
  }
  __TOC__(centernet_pre)

  __TIC__(CenterNet_setimg)
  configurable_dpu_task_->setInputImageRGB(vimg);
  __TOC__(CenterNet_setimg)
  __TIC__(CenterNet_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(CenterNet_dpu)

  __TIC__(CenterNet_post)
  auto results = centernet_post_process();
  __TOC__(CenterNet_post)

  __TOC__(CenterNet_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
