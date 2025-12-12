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
#ifndef DEEPHI_Bosch_Fcnsemseg_HPP_
#define DEEPHI_Bosch_Fcnsemseg_HPP_

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/bosch_fcnsemseg.hpp>

using std::shared_ptr;
using std::vector;

namespace vitis {

namespace ai {

class Bosch_FcnsemsegImp : public vitis::ai::TConfigurableDpuTask<Bosch_Fcnsemseg> {
 public:
  Bosch_FcnsemsegImp(const std::string &model_name, bool need_preprocess = true);
  virtual ~Bosch_FcnsemsegImp();

 private:

  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;

  virtual Bosch_FcnsemsegResult run(const cv::Mat &img) override;
  virtual std::vector<Bosch_FcnsemsegResult> run( const std::vector<cv::Mat> &img) override;
  std::vector<Bosch_FcnsemsegResult> bosch_fcnsemseg_post_process();
  Bosch_FcnsemsegResult bosch_fcnsemseg_post_process(int idx);
  void pad_or_crop(cv::Mat img, int height,int width, int idx) ;
  void preprocess(cv::Mat img, int idx) ;
  void put_data_inputlayer (cv::Mat img, int8_t* dest) ;
  
  int batch_size = 1;
  int real_batch_size = 1;
  int sWidth;
  int sHeight;
  int sChannel[2];
  float sScaleo[2];
  int o_idx[2];
  std::vector<float> mean, scale;
  float inScale =0;
  int inWidth, inHeight, inChannel;

  std::vector<std::vector<float>> vexp;
  std::vector<std::vector<int8_t*>> sData;
  std::vector<float> anchors;
  std::vector<int8_t*> in_addr;

  int ENABLE_BOSCHFCNSEMSEG_DEBUG = 0;
};
}  // namespace ai
}  // namespace vitis

#endif
