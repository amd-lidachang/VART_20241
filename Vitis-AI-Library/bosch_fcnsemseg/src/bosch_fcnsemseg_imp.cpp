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
#include "./bosch_fcnsemseg_imp.hpp"

#include <fstream>
#include <iostream>
#include <numeric>
#include <sys/stat.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

DEF_ENV_PARAM(ENABLE_BOSCHFCNSEMSEG_DEBUG, "0");
DEF_ENV_PARAM(ENABLE_BOSCHFCNSEMSEG_IMPORTINPUT, "0");

template<typename T>
void myreadfile(T* dest, int size1, const std::string& filename)
{
  ifstream Tin;
  Tin.open(filename, ios_base::in|ios_base::binary);
  if(!Tin)  {
     cout<<"Can't open the file! " << filename << std::endl;
     return;
  }
  Tin.read( (char*)dest, size1*sizeof(T));
}

template void myreadfile(float*dest, int size1, const std::string& filename);
template void myreadfile(int8_t*dest, int size1, const std::string& filename);

int getfloatfilelen(const std::string& file)
{
  struct stat statbuf;
  if(stat(file.c_str(), &statbuf)!=0){
    std::cerr << " bad file stat " << file << std::endl;
    exit(-1);
  }
  return statbuf.st_size/4;
}

void clip_data(int itype, float* src_f, signed char*dst_c, int num, float scale)
{
  if (itype == 1) {
     scale = 1.0/scale;
  }
  for(int i=0; i<num; i++) {
    if (src_f[i] > 127*scale ) {
       dst_c[i]=127;
    } else if (src_f[i] < -128*scale ){
       dst_c[i]=-128;  // std::cout <<" smaller for arm_loc!\n";
    } else {
      dst_c[i] = (signed char)( round(src_f[i]/scale));
    }
  }
}

// 0:out;  1:in
void import_data(int itype, const std::string& filename, int8_t* dst_addr, float scale) {
    int len = getfloatfilelen( filename);
    float* fbuf=new float[ len ];
    myreadfile( fbuf, len, filename);
    clip_data(itype, fbuf, dst_addr, len, scale ) ;
    delete []fbuf;
}

static vector<float> get_means(const vitis::ai::proto::DpuKernelParam& c) {
  return vector<float>(c.mean().begin(), c.mean().end());
}

static vector<float> get_scales(const vitis::ai::proto::DpuKernelParam& c) {
  return vector<float>(c.scale().begin(), c.scale().end());
}

Bosch_FcnsemsegImp::Bosch_FcnsemsegImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<Bosch_Fcnsemseg>(model_name, need_preprocess),
      input_tensors_ (configurable_dpu_task_->getInputTensor()[0]),
      output_tensors_(configurable_dpu_task_->getOutputTensor()[0])
{
  ENABLE_BOSCHFCNSEMSEG_DEBUG = ENV_PARAM(ENABLE_BOSCHFCNSEMSEG_DEBUG);

  batch_size = get_input_batch();
  sData.resize(batch_size);
  for(int i=0; i<batch_size; i++) {
    sData[i].resize(2);
  }

  for(int i=0; i<2; i++) {
    for(int j=0; j<batch_size; j++) {
       sData[j][i] =(int8_t*) output_tensors_[i].get_data(j);
    }
    sWidth = output_tensors_[i].width;
    sHeight= output_tensors_[i].height;
    sChannel[i] = output_tensors_[i].channel;
    sScaleo[i] =  tensor_scale(output_tensors_[i]);
  }
  // net0out: width height channel scale :  480  272  1  0.125
  // net0out: width height channel scale :  480  272  9  0.125

  inScale =  tensor_scale(input_tensors_[0]);
  inWidth = input_tensors_[0].width;
  inHeight= input_tensors_[0].height;
  inChannel = input_tensors_[0].channel;
  if (ENABLE_BOSCHFCNSEMSEG_DEBUG ) {
      std::cout <<"net0out: width height channel scale :  " << sWidth << "  " << sHeight << "  " <<  sChannel[0] <<  "  " << sScaleo[0] << "    " << sChannel[1] << "  " << sScaleo[1] << "\n";
      std::cout <<"netin: width height channel scale :  " << inWidth << "  " << inHeight << "  " <<  inChannel   <<  "  " << inScale << "\n";
      // 1920  1088  3  16
  }

  //  272 480 1   1 272 480 9
  for(int i=0;i<2;i++){
    if (sChannel[i] == 9) { o_idx[0] = i; continue; }
    if (sChannel[i] == 1) { o_idx[1] = i; continue; }
  }
  std::vector<std::vector<float>> vexp1(sWidth * sHeight , std::vector<float>(9,0.0));
  vexp.swap(vexp1);

  mean  = get_means (configurable_dpu_task_->getConfig().kernel(0));
  scale = get_scales(configurable_dpu_task_->getConfig().kernel(0));

  for(int i=0; i<batch_size; i++) {
    in_addr.emplace_back( (int8_t*)input_tensors_[0].get_data(i));
    memset(in_addr[i], 0, inWidth*inHeight*inChannel);
  }
}

Bosch_FcnsemsegImp::~Bosch_FcnsemsegImp() {}

std::vector<Bosch_FcnsemsegResult> Bosch_FcnsemsegImp::bosch_fcnsemseg_post_process() {

  auto ret = std::vector<vitis::ai::Bosch_FcnsemsegResult>{};
  ret.reserve(real_batch_size);
  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(bosch_fcnsemseg_post_process(i));
  }
  return ret;
}

Bosch_FcnsemsegResult Bosch_FcnsemsegImp::bosch_fcnsemseg_post_process(int idx) {

#if 0  
  // ignore exp logic for segmentation
  
  // sub  exp reduction_sum div   for ( 9, 1) . 
  __TIC__(exp)
  std::vector<float> tmpf(9, 0.0);
  for(int i=0; i<sHeight*sWidth; i++) {
     for (int j=0;j<9;j++) {
       tmpf[j] = exp(sScaleo[o_idx[0]]*sData[idx][o_idx[0]][i*9+j] - sScaleo[o_idx[1]]*sData[idx][o_idx[1]][i]);
     }
     float sum = std::accumulate(tmpf.begin(), tmpf.end(), 0.0);
     for (int j=0;j<9;j++) {
       vexp[i][j] = tmpf[j]/sum;
       std::cout << vexp[i][j]  << " ";
     } std::cout <<"\n";
  }
  __TOC__(exp)
#endif
  
  cv::Mat rmat(sHeight, sWidth, CV_8UC1);
  int L0 = o_idx[0];
  int col_ind=0, row_ind=0;
  for(int i=0; i<sHeight*sWidth*sChannel[L0] ; i+=sChannel[L0]) {
    auto max_ind = std::max_element( sData[idx][L0] + i,
                                     sData[idx][L0] + i + sChannel[L0] );
    // for(int j=0;j<9;j++){ std::cout << int(*(int8_t*)(sData[idx][o_idx[0]] + i +j)) << " ";} std::cout <<"\n";
    uint8_t posit = std::distance( sData[idx][L0] + i, max_ind);
    rmat.at<uchar>(row_ind, col_ind) = posit;
    // rmat.at<uchar>(row_ind, col_ind) = posit*30;
    // std::cout << row_ind << " "<< col_ind << "  " << ((int)posit)*30 <<"\n";
    col_ind++;
    if (col_ind > sWidth - 1) {
      row_ind++;
      col_ind = 0;
    }
  }

  return Bosch_FcnsemsegResult{int(input_tensors_[0].width), int(input_tensors_[0].height), rmat};
}

void Bosch_FcnsemsegImp::preprocess(cv::Mat img, int idx) {

  // normalize
  int rows1 = img.rows;
  int cols1 = img.cols;
  uint8_t* input = img.data;
  
  __TIC__(pre_norm)
  cv::Mat omat(rows1, cols1, CV_32FC3 );

  //std::cout <<"inmat continus : " << img.isContinuous() <<" " << rows1 << " " << cols1 << " " << img.step << " " ;  // 1 1 6780 6780=565*4*3 
  //std::cout <<"  out : " <<  omat.isContinuous() << " " << omat.step << " " << omat.step[0] << "\n";  // 1 1 6780 6780=565*4*3 
  float* dest = (float*)omat.data;
  int channels=3;
  int step = img.step;
  // std::cout <<" img r c vs dpu r c : " << rows1 << " " << cols1 << " " << rows << " " << cols << "\n";
  for (auto h = 0; h < rows1; ++h) {
    for (auto w = 0; w < cols1; ++w) {
      for (auto c = 0; c < channels; ++c) {
        float value = (input[h * step + w * channels + c] * 1.0f - mean[c]) * scale[c];
        // std::cout << value << " ";
        dest[h * cols1 * channels + w * channels + abs(c - 2)] = value;
        // if (h==0 && w==0) std::cout<<"value:" << c << " " << mean[c] << "  " << scale[c] << "  " << value << "\n";
      }
    } 
  }
  __TOC__(pre_norm)

  // pad_or_crop
  pad_or_crop(omat, 480, 640, idx);
}

void Bosch_FcnsemsegImp::pad_or_crop(cv::Mat img, int height,int width, int idx) {

  __TIC__(pre_padcrop)
  cv::Mat  tmp(img);

  int diff0 = height - img.rows;
  int diff1 = width - img.cols;
  int crop0 = std::max(-diff0/2, 0);
  int crop1 = std::max(-diff1/2, 0);
  int crop[2][2];
  crop[0][0] = crop0;
  crop[0][1] = std::max(-diff0-crop0, 0);
  crop[1][0] = crop1;
  crop[1][1] = std::max(-diff1-crop1, 0);

  // std::cout <<" crop : " << crop[0][0] << " " << crop[0][1] << " " << crop[1][0] << " " << crop[1][1]  <<"\n";
  if (crop[0][0] || crop[0][1] || crop[1][0] || crop[1][1] ) {
     crop[0][1] = img.rows - crop[0][1];
     crop[1][1] = img.cols - crop[1][1];
     // std::cout <<" crop : " << crop[0][0] << " " << crop[0][1] << " " << crop[1][0] << " " << crop[1][1]  <<"\n";  // 80 560 0 565
     tmp = img(cv::Rect(crop[1][0],  crop[0][0], crop[1][1]-crop[1][0] , crop[0][1]-crop[0][0]));

  }
  cv::Mat img2(tmp);

  int pad0 = std::max(diff0/2, 0);
  int pad1 = std::max(diff1/2, 0);
  int pad[2][2];
  pad[0][0] = pad0; 
  pad[0][1] = std::max( diff0-pad0 ,0);
  pad[1][0] = pad1; 
  pad[1][1] = std::max( diff1-pad1 ,0);

  // std::cout <<" pad:" <<  pad[0][0] << " " << pad[0][1] << " " << pad[1][0] << " " << pad[1][1] <<"\n"; // pad:0 0 37 38
  if (pad[0][0] || pad[0][1] || pad[1][0] || pad[1][1] ) {

    //  void cv::copyMakeBorder	( InputArray 	src,
    //                            OutputArray 	dst,
    //                            int 	top,
    //                            int 	bottom,
    //                            int 	left,
    //                            int 	right,
    //                            int 	borderType,
    //                            const Scalar & 	value = Scalar() 
    //  )	
    // std::cout <<"tmp-wh :" << tmp.cols << " " << tmp.rows << "   destimg-wh : " << img.cols << " " << img.cols << "\n";  // tmp-wh :565 480   destimg-wh : 1205 1205
    cv::copyMakeBorder(tmp, 
                       img2, 
                       pad[0][0],
                       pad[0][1],
                       pad[1][0], 
                       pad[1][1], 
                       cv::BORDER_CONSTANT, 
                       0  );
    // std::cout <<"tmp-wh :" << tmp.cols << " " << tmp.rows << "   destimg-wh : " << img2.cols << " " << img2.rows << "\n";  // tmp-wh :565 480   destimg-wh : 1205 1205
    if (0) {
       float* input = (float*)img.data;
       for (auto h = 0; h < img.rows; ++h) {
         for (auto w = 0; w < img.cols; ++w) {
           for (auto c = 0; c < 3; ++c) {
             float value = input[h * img.cols * 3 + w * 3 + c] ; (void)value; std::cout << value << " "; 
           }
         }   std::cout <<"\n";
       }   
    }
  }
  __TOC__(pre_padcrop)

  __TIC__(pre_resize)
  cv::Mat ret(getInputHeight()-8, getInputWidth(), CV_32FC3, cv::Scalar(0.0, 0.0, 0.0));
  cv::resize(img2, ret, cv::Size(getInputWidth(), getInputHeight() - 8), 0, 0, cv::INTER_LINEAR);
  __TOC__(pre_resize)
  if (0) {
     float* input = (float*)img.data;
     for (auto h = 0; h < ret.rows; ++h) {
       for (auto w = 0; w < ret.cols; ++w) {
         for (auto c = 0; c < 3; ++c) {
            float value = input[h * ret.cols * 3 + w * 3 + c] ; std::cout << value << " "; 
         }
       } 
       std::cout <<"\n";
     }   
  }

  __TIC__(pre_put_data)
  put_data_inputlayer(ret, in_addr[idx]); 
  __TOC__(pre_put_data)
}

void Bosch_FcnsemsegImp::put_data_inputlayer (cv::Mat img, int8_t* dest) {
  int rows1 = img.rows;
  int cols1 = img.cols;
  float* input = (float*)img.data;
  int channels=3;

  for (auto h = 0; h < rows1; ++h) {
    for (auto w = 0; w < cols1; ++w) {
      for (auto c = 0; c < channels; ++c) {
        int value = round(input[h * cols1 * channels + w * channels + c] * inScale);
        //if (value >127 || value <=-128 )  std::cout << value << "   " << h << " " << w << " " << c << "\n";
        dest[h * cols1 * channels + w * channels + c] = value;
      }
    }
  }
}

Bosch_FcnsemsegResult Bosch_FcnsemsegImp::run( const cv::Mat &input_img) {
  real_batch_size = 1;

  __TIC__(Bosch_Fcnsemseg_total)
  __TIC__(Bosch_Fcnsemseg_setimg)
  preprocess(input_img, 0);

  if (ENV_PARAM(ENABLE_BOSCHFCNSEMSEG_IMPORTINPUT) == 1) {
    import_data(1, "./inputlayer_fcnsemseg.bin", (int8_t*)input_tensors_[0].get_data(0) , inScale);
  }


  __TOC__(Bosch_Fcnsemseg_setimg)
  __TIC__(Bosch_Fcnsemseg_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(Bosch_Fcnsemseg_dpu)

  __TIC__(Bosch_Fcnsemseg_post)
  auto results = bosch_fcnsemseg_post_process();
  __TOC__(Bosch_Fcnsemseg_post)

  __TOC__(Bosch_Fcnsemseg_total)
  return results[0];
}

std::vector<Bosch_FcnsemsegResult> Bosch_FcnsemsegImp::run( const std::vector<cv::Mat> &input_img) {
  __TIC__(Bosch_Fcnsemseg_total)
  __TIC__(Bosch_Fcnsemseg_setimg)
  real_batch_size = std::min(int(input_img.size()), int(batch_size));
  for (auto i = 0; i < real_batch_size; i++) {
     preprocess(input_img[i], i);
  }

  __TOC__(Bosch_Fcnsemseg_setimg)
  __TIC__(Bosch_Fcnsemseg_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(Bosch_Fcnsemseg_dpu)

  __TIC__(Bosch_Fcnsemseg_post)
  auto results = bosch_fcnsemseg_post_process();
  __TOC__(Bosch_Fcnsemseg_post)

  __TOC__(Bosch_Fcnsemseg_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
