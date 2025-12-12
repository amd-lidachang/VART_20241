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
/*
 * Filename: bts.hpp
 *
 * Description:
 * This network is used to getting depth of input images
 * image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include "vitis/ai/configurable_dpu_task.hpp"

namespace vitis {
namespace ai {

struct BTSResult {
  int width;
  int height;
  /// Mat data format: CV_16U
  cv::Mat depth_mat;
};

class BTS : public ConfigurableDpuTaskBase {
 public:
  /**
 * @brief Factory function to get instance of derived classes of class
 * BTS
 *
 * @param model_name Model name
 * @param need_preprocess Normalize with mean/scale or not, default
 value is true.
 * @return An instance of BTS class.
 */
  static std::unique_ptr<BTS> create(const std::string& model_name,
                                     bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit BTS(const std::string& model_name, bool need_preprocess);
  BTS(const BTS&) = delete;
  BTS& operator=(const BTS&) = delete;

 public:
  virtual ~BTS();
  /**
   * @brief Function to get running result of the facedetect network.
   *
   * @param img Input Data ,input image (cv::Mat) need to be resized to
   *InputWidth and InputHeight required by the network.
   *
   * @return BTSResult
   *
   */
  virtual BTSResult run(const cv::Mat& img) = 0;

  /**
   * @brief Function to get running results of the facedetect neural network in
   * batch mode.
   *
   * @param imgs Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch. The input
   * images need to be resized to InputWidth and InputHeight required by the
   * network.
   *
   * @return The vector of BTSResult.
   *
   */
  virtual std::vector<BTSResult> run(const std::vector<cv::Mat>& imgs) = 0;
};

}  // namespace ai
}  // namespace vitis

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: utf-8-unix
// End:
