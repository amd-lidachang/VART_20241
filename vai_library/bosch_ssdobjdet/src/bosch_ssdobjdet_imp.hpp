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
#ifndef DEEPHI_Bosch_Ssdobjdet_HPP_
#define DEEPHI_Bosch_Ssdobjdet_HPP_

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/bosch_ssdobjdet.hpp>

using std::shared_ptr;
using std::vector;

namespace vitis {

namespace ai {

class Bosch_SsdobjdetImp : public vitis::ai::TConfigurableDpuTask<Bosch_Ssdobjdet> {
 public:
  Bosch_SsdobjdetImp(const std::string &model_name, bool need_preprocess = true);
  virtual ~Bosch_SsdobjdetImp();

 private:

  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;

  virtual Bosch_SsdobjdetResult run(const cv::Mat &img) override;
  virtual std::vector<Bosch_SsdobjdetResult> run( const std::vector<cv::Mat> &img) override;
  std::vector<Bosch_SsdobjdetResult> bosch_ssdobjdet_post_process();
  Bosch_SsdobjdetResult bosch_ssdobjdet_post_process(int idx);
  std::vector<float> decode_box(int i, int idx) ;

  int batch_size = 1;
  int real_batch_size = 1;
  int ENABLE_BOSCHOBJDET_DEBUG = 0;
  int sWidth[3];
  int sHeight;
  int sChannel;
  float sScaleo[3];
  int o_idx[3];
  std::vector<std::vector<int8_t*>> sData;
  int8_t* insData;
  std::vector<float> anchors;

  float score_threshold = 0.001;
  float nms_thresh = 0.7;
  int top_k = 125;

};
}  // namespace ai
}  // namespace vitis

#endif
