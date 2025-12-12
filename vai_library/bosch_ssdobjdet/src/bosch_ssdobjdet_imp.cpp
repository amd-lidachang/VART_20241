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
#include "./bosch_ssdobjdet_imp.hpp"
#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/nnpp/apply_nms.hpp>

using namespace std;
namespace vitis {
namespace ai {

DEF_ENV_PARAM(ENABLE_BOSCHOBJDET_DEBUG, "0");
DEF_ENV_PARAM(ENABLE_BOSCHOBJDET_IMPORTIN, "0");
DEF_ENV_PARAM(ENABLE_BOSCHOBJDET_IMPORTOUT, "0");
DEF_ENV_PARAM(ENABLE_BOSCHOBJDET_FIXANCHOR, "1");
DEF_ENV_PARAM_2(BOSCHOBJDET_CONFIDENCE,   "0.001", float);

int getfloatfilelen(const std::string& file)
{
  struct stat statbuf;
  if(stat(file.c_str(), &statbuf)!=0){
    std::cerr << " bad file stat " << file << std::endl;
    exit(-1);
  }
  return statbuf.st_size/4;
}

template<typename T>
void myreadfile(T* dest, const std::string& filename)
{
  ifstream Tin;
  Tin.open(filename, ios_base::in|ios_base::binary);
  if(!Tin)  {
     cout<<"Can't open the file! " << filename << std::endl;
     return;
  }
  int size1 = getfloatfilelen(filename);
  Tin.read( (char*)dest, size1*sizeof(float));
}
  
void myapplyNMS( std::vector<std::vector<float>>& boxes, 
                 std::vector<float>& scores, 
                 float nms_thresh, 
                 int top_k,
                 std::vector<size_t>& keep)
{
  const size_t count = boxes.size();
  vector<pair<float, size_t>> order;
  for (size_t i = 0; i < count; ++i) {
    order.push_back({scores[i], i});
  }
  std::sort(order.begin(), order.end(),
         [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
           return ls.first > rs.first;
         });
  vector<int> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
            [](auto& km) { return km.second; });

  std::vector<float> area(boxes.size(), 0);
  
  auto get_area = [&](int idx) { 
     if (area[idx] == 0 ) { 
       area[idx] = (boxes[idx][2]-boxes[idx][0])*(boxes[idx][3]-boxes[idx][1]);
     }
     return area[idx];
  };
  
  // for(int i=0; i< (int)boxes.size(); i++) {
  //   area[i] =  (boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1]);
  // }

  std::vector<bool> suppressed_rw(count, false); // already removed

  float inter_upleft_0, inter_upleft_1;
  float inter_botright_2, inter_botright_3;
  float inter_wh_0, inter_wh_1;
  float inter, min_area, iou = 0.0;
  for (size_t _i = 0; _i < count; ++_i) {
    size_t i = ordered[_i];  

    if (suppressed_rw[i]) {  
      continue; 
    }
    keep.push_back(i);
    if ((int)keep.size() == top_k) {
      break;
    }
    for (size_t _j = _i + 1; _j < count; ++_j) {
      size_t j = ordered[_j];
      if (suppressed_rw[j]) { 
        continue; 
      }
      inter_upleft_0   = std::max(boxes[i][0], boxes[j][0]);
      inter_upleft_1   = std::max(boxes[i][1], boxes[j][1]);
      inter_botright_2 = std::min(boxes[i][2], boxes[j][2]);
      inter_botright_3 = std::min(boxes[i][3], boxes[j][3]);

      inter_wh_0 = std::max(0.0f, inter_botright_2 - inter_upleft_0);
      inter_wh_1 = std::max(0.0f, inter_botright_3 - inter_upleft_1);

      inter = inter_wh_0 * inter_wh_1;
      // min_area = std::min( area[i], area[j] );
      min_area = std::min( get_area(i), get_area(j) );

      if ( !(std::abs(min_area) < 0.001f )) {
         iou = inter/min_area;
         if (iou >nms_thresh) { 
           suppressed_rw[j] = true;
         }
      }
    }
  }
}

Bosch_SsdobjdetImp::Bosch_SsdobjdetImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<Bosch_Ssdobjdet>(model_name, need_preprocess),
      input_tensors_ (configurable_dpu_task_->getInputTensor()[0]),
      output_tensors_(configurable_dpu_task_->getOutputTensor()[0])
{
  ENABLE_BOSCHOBJDET_DEBUG = ENV_PARAM(ENABLE_BOSCHOBJDET_DEBUG);
  score_threshold = ENV_PARAM(BOSCHOBJDET_CONFIDENCE);
  // std::cout <<" score_threshold :" << score_threshold <<"\n";

  std::string model_namex(model_name);
  if (model_name.size() > 7 && model_name.substr( model_name.size()-7, 7) == ".xmodel") {
     size_t pos = 0;
     if ((pos = model_name.rfind("/")) != std::string::npos) {
        model_namex = model_name.substr(pos+1, model_name.size()-7-(pos+1) );
     } else {
        model_namex = model_name.substr(0, model_name.size()-7);
     }
  }

  std::string anchors_bin = std::string(configurable_dpu_task_->get_graph()->get_attr<std::string>("dirname")) + "/" + model_namex + "_anchors.bin";

  int len = getfloatfilelen(anchors_bin);
  anchors.resize(len);
  myreadfile(anchors.data(), anchors_bin);
  if (0) { 
    std::cout <<"anchors_bin :" << anchors_bin <<"\n";
    for(int i=0; i<len; i++) { std::cout << anchors[i] << " ";  if ((i+1)%4==0) std::cout <<"\n"; }
  }

  batch_size = get_input_batch();
  sData.resize(batch_size);
  for(int i=0; i<batch_size; i++) {
    sData[i].resize(3);
  }

  insData = (int8_t*)input_tensors_[0].get_data(0);

  for(int i=0; i<3; i++) {
    for(int j=0; j<batch_size; j++) {
       sData[j][i] =(int8_t*) output_tensors_[i].get_data(j);
    }
    sWidth[i] = output_tensors_[i].width;
    sHeight= output_tensors_[i].height;
    sChannel = output_tensors_[i].channel;
    sScaleo[i] =  tensor_scale(output_tensors_[i]);
    if (ENABLE_BOSCHOBJDET_DEBUG ) {
      std::cout <<"net0out: width height channel scale :  " << sWidth[i] << "  " << sHeight << "  " <<  sChannel <<  "  " << sScaleo[i] << "\n";
    }
  }
  //  net0out: sWidth heiht channel size  scale :  4  170040  1  680160  0.0625
  //  net0out: sWidth heiht channel size  scale :  1  170040  1  170040  1
  //  net0out: sWidth heiht channel size  scale :  2  170040  1  340080  1
  //  in model:  sequence is 4 2 1 
  for(int i=0;i<3;i++){
    if (sWidth[i] == 4) { o_idx[0] = i; continue; }
    if (sWidth[i] == 2) { o_idx[1] = i; continue; }
    if (sWidth[i] == 1) { o_idx[2] = i; continue; }
  }
}

Bosch_SsdobjdetImp::~Bosch_SsdobjdetImp() {}

std::vector<Bosch_SsdobjdetResult> Bosch_SsdobjdetImp::bosch_ssdobjdet_post_process() {

  auto ret = std::vector<vitis::ai::Bosch_SsdobjdetResult>{};
  ret.reserve(real_batch_size);
  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(bosch_ssdobjdet_post_process(i));
  }
  return ret;
}

std::vector<float> Bosch_SsdobjdetImp::decode_box(int i, int idx) {
  vector<float> bbox(4, 0);
  int L0 = o_idx[0];
  // here, we get center_x, center_y, w, h 
#if 0
  bbox[0] = (    sData[idx][L0][ i*4+0]*sScaleo[L0]) * anchors[i*4+2] + anchors[i*4+0];
  bbox[1] = (    sData[idx][L0][ i*4+1]*sScaleo[L0]) * anchors[i*4+3] + anchors[i*4+1];
  bbox[2] = exp( sData[idx][L0][ i*4+2]*sScaleo[L0]) * anchors[i*4+2];
  bbox[3] = exp( sData[idx][L0][ i*4+3]*sScaleo[L0]) * anchors[i*4+3];
#else
  float fix_xy=1.0, fix_wh=1.0;
  if (ENV_PARAM(ENABLE_BOSCHOBJDET_FIXANCHOR) == 1) {
     fix_xy = 0.1;
     fix_wh = 0.2;
  }
  bbox[0] = (    sData[idx][L0][ i*4+0]*sScaleo[L0] * fix_xy) * anchors[i*4+2] + anchors[i*4+0];
  bbox[1] = (    sData[idx][L0][ i*4+1]*sScaleo[L0] * fix_xy) * anchors[i*4+3] + anchors[i*4+1];
  bbox[2] = exp( sData[idx][L0][ i*4+2]*sScaleo[L0] * fix_wh) * anchors[i*4+2];
  bbox[3] = exp( sData[idx][L0][ i*4+3]*sScaleo[L0] * fix_wh) * anchors[i*4+3];
#endif

#if 1
  // here, we get x1 y1 , x2 y2 for topleft & bottomright
  // this is need by new version of nms function
  bbox[0] = bbox[0]-bbox[2]*0.5;
  bbox[1] = bbox[1]-bbox[3]*0.5;
  bbox[2] = bbox[0]+bbox[2]; // bad. should be no *0.5
  bbox[3] = bbox[1]+bbox[3]; // bad. should be no *0.5
#endif

  //for (int j=0;j<4;j++) { std::cout << " " <<sData[idx][L0][ i*4+j]*sScaleo[L0] << " " ; } 
  //std::cout <<"anchors :"; for(int j=0;j<4;j++){std::cout<< anchors[i*4+j] <<" ";} std::cout <<"\n";
  //std::cout <<"decode: " << bbox[0] << " " << bbox[1] << " " << bbox[2] << " " << bbox[3] <<"\n";
  return bbox;
}

Bosch_SsdobjdetResult Bosch_SsdobjdetImp::bosch_ssdobjdet_post_process(int idx) {

  Bosch_SsdobjdetResult  ret{int(input_tensors_[0].width), int(input_tensors_[0].height)};

  if (ENV_PARAM(ENABLE_BOSCHOBJDET_IMPORTOUT) == 1) {
    myreadfile( sData[0][o_idx[0]], "/home/root/pp/ssdoutputc4.bin");
    myreadfile( sData[0][o_idx[1]], "/home/root/pp/ssdoutputc2.bin");
  }

  // sub  exp reduction_sum div   for ( 2, 1) .  then concat 4 with result
  __TIC__(exp_decode)
  std::vector<std::vector<float>> boxes;
  std::vector<float> scores;
  float exp1 = 0.0;
  int L1 = o_idx[1], L2 = o_idx[2]; (void)L2;
  float score_threshold_fix = log(1.0/score_threshold-1);  (void)score_threshold_fix;

#if 1
  // full version;
  for(int i=0; i<sHeight; i++) {
     auto tmp1 = exp(sScaleo[L1]*sData[idx][L1][i*2+0] - sScaleo[L2]*sData[idx][L2][i]);
     auto tmp2 = exp(sScaleo[L1]*sData[idx][L1][i*2+1] - sScaleo[L2]*sData[idx][L2][i]);
     exp1 = tmp2/(tmp1+tmp2);

    // std::cout << sScaleo[L1]*sData[idx][L1][i*2+0] - sScaleo[L2]*sData[idx][L2][i] << " " << sScaleo[L1]*sData[idx][L1][i*2+1] - sScaleo[L2]*sData[idx][L2][i] <<"\n";
     // std::cout << sScaleo[L1]*sData[idx][L1][i*2+0] << " " << sScaleo[L1]*sData[idx][L1][i*2+1] <<"\n";
     if (exp1 > score_threshold) {
       boxes.emplace_back(decode_box(i, idx));
       scores.emplace_back( exp1 );
     }
  }
#else
  // optimized version
  for(int i=0; i<sHeight; i++) {
     auto tmp1 = sScaleo[L1]*sData[idx][L1][i*2+0] ;
     auto tmp2 = sScaleo[L1]*sData[idx][L1][i*2+1] ;
     exp1 = tmp1 - tmp2;
     if (exp1 <= score_threshold_fix) {
       boxes.emplace_back(decode_box(i, idx));
       scores.emplace_back( exp(tmp2)/(exp(tmp1)+exp(tmp2)) );
     }
     // std::cout <<i << "  : " << vexp[i*2+0] << " " << vexp[i*2+1]  <<"\n";
  }
#endif
  // std::cout <<"score :" << score_threshold << " " << score_threshold_fix << " " << boxes.size() <<"\n";
  __TOC__(exp_decode)

  // next postprocess from bosch

  // 2. nms
  __TIC__(nms)
  std::vector<size_t> results;
#if 0
  // use nms function in xnnpp; not correct in this case.
  applyNMS(boxes, scores, nms_thresh, score_threshold, results);
#else
  myapplyNMS(boxes, scores, nms_thresh, top_k, results);
#endif

  __TOC__(nms)
  __TIC__(make_result)

  // only need in applyNMS(); 
  // if ((int) results.size() > top_k ) { 
  //    results.resize(top_k); 
  // }
  for (auto& it: results) {
    Bosch_SsdobjdetResult::BoundingBox res;
    res.label = 1;  // fake value;
    res.score = scores[it];

    auto& bbox = boxes[it];  (void)bbox;
    // we need leftX, leftY, w, h 
#if 0
    // if use applyNMS()
    res.x = std::max(0.0, bbox[0] - bbox[2] / 2.0);
    res.y = std::max(0.0, bbox[1] - bbox[3] / 2.0);
    res.width  = bbox[2];
    res.height = bbox[3];
#else
    // use myapplyNMS
    res.x = bbox[0];
    res.y = bbox[1];
    res.width  = bbox[2]-bbox[0];
    res.height = bbox[3]-bbox[1];
#endif
    // std::cout <<"result : " <<res.x << " " << res.y << " " << res.width << " " << res.height << " " << res.score << "\n";
    ret.bboxes.emplace_back(res);
  }
  __TOC__(make_result)

  return ret;
}

Bosch_SsdobjdetResult Bosch_SsdobjdetImp::run( const cv::Mat &input_img) {
  cv::Mat img;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_img.size()) {
    cv::resize(input_img, img, size, 0, 0, cv::INTER_LINEAR);
  } else {
    img = input_img;
  }
  __TIC__(Bosch_Ssdobjdet_total)
  __TIC__(Bosch_Ssdobjdet_setimg)
  real_batch_size = 1;
  configurable_dpu_task_->setInputImageRGB(img);

  if (ENV_PARAM(ENABLE_BOSCHOBJDET_IMPORTIN) == 1) {
    myreadfile( insData, "/home/root/pp/ssdinput.bin");
  }
  __TOC__(Bosch_Ssdobjdet_setimg)
  __TIC__(Bosch_Ssdobjdet_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(Bosch_Ssdobjdet_dpu)

  __TIC__(Bosch_Ssdobjdet_post)
  auto results = bosch_ssdobjdet_post_process();
  __TOC__(Bosch_Ssdobjdet_post)

  __TOC__(Bosch_Ssdobjdet_total)
  return results[0];
}

std::vector<Bosch_SsdobjdetResult> Bosch_SsdobjdetImp::run( const std::vector<cv::Mat> &input_img) {
  auto size = cv::Size(getInputWidth(), getInputHeight());
  real_batch_size = std::min(int(input_img.size()), int(batch_size));
  std::vector<cv::Mat> vimg(real_batch_size);
  for (auto i = 0; i < real_batch_size; i++) {
    if (size != input_img[i].size()) {
      cv::resize(input_img[i], vimg[i], size, 0, 0, cv::INTER_LINEAR);
    } else {
      vimg[i] = input_img[i];
    }
  }
  __TIC__(Bosch_Ssdobjdet_total)
  __TIC__(Bosch_Ssdobjdet_setimg)

  configurable_dpu_task_->setInputImageRGB(vimg);

  __TOC__(Bosch_Ssdobjdet_setimg)
  __TIC__(Bosch_Ssdobjdet_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(Bosch_Ssdobjdet_dpu)

  __TIC__(Bosch_Ssdobjdet_post)
  auto results = bosch_ssdobjdet_post_process();
  __TOC__(Bosch_Ssdobjdet_post)

  __TOC__(Bosch_Ssdobjdet_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
