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
#ifndef DEEPHI_CenterNet_HPP_
#define DEEPHI_CenterNet_HPP_

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/centernet.hpp>


#define TESTF 0

using std::shared_ptr;
using std::vector;

namespace vitis {

namespace ai {

class CenterNetImp : public vitis::ai::TConfigurableDpuTask<CenterNet> {
 public:
  CenterNetImp(const std::string &model_name, bool need_preprocess = true);
  virtual ~CenterNetImp();

 private:

  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;

  virtual CenterNetResult run(const cv::Mat &img) override;
  virtual std::vector<CenterNetResult> run( const std::vector<cv::Mat> &img) override;
  std::vector<CenterNetResult> centernet_post_process();

  cv::Mat preprocess(const cv::Mat img, int idx) ;
  cv::Mat get_affine_transform(float center_x, float center_y, float scale, int o_w, int o_h, bool inv); 
  CenterNetResult centernet_post_process(int idx);

  int real_batch_size = 1;
  int batch_size = 1 ;

  std::vector<int> img_h, img_w, img_maxhw;

  int iwidth, iheight;
  int owidth, oheight;
  const int num_classes = 80;
  float score_threshold = 0.1;
  float score_threshold_f = 0.1;
#if TESTF
  std::vector<float*> oData_hm;
  std::vector<float*> oData_wh;
  std::vector<float*> oData_reg;
#else
  std::vector<int8_t*> oData_hm;
  std::vector<int8_t*> oData_wh;
  std::vector<int8_t*> oData_reg;
#endif
  float oscale_hm, oscale_wh, oscale_reg;
 
  int idx_hm, idx_wh, idx_reg;
  int pad = 1;
  int topk = 100;

};
}  // namespace ai
}  // namespace vitis

#endif
