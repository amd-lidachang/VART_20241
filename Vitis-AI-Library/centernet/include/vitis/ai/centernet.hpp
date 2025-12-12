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
 * Filename: CenterNet.hpp
 *
 * Description:
 * This network is used to classify the object from a input image.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <array>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {

struct CenterNetResult{
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  struct Det_Res{
    float score;
    std::array<float, 4> pos;
    Det_Res(float score1, std::array<float,4> pos1) : score(score1), pos(pos1){
      // memcpy(pos, pos1, sizeof(float)*sizeof(pos));
    }
  };
  // struct Cls_Res {
  //    int label;
  //    std::vector<Det_Res> vcls;
  // };
  // std::vector<Cls_Res> vres;
  std::vector<std::vector<Det_Res>> vres;
};

/**
 * @brief Base class for CenterNet (production recognication)
 *
 * Input is an image (cv:Mat).
 *
 * Output is a struct of classification results, named CenterNetResult.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_CenterNet.jpg");
   auto CenterNet = vitis::ai::CenterNet::create("centernet_pt",true);
   auto result = CenterNet->run(img);
   // result is structure holding the classindex .
   std::cout << result.classidx <<"\n";
   @endcode
 *
 */
class CenterNet {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * CenterNet.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of CenterNet class.
   *
   */
  static std::unique_ptr<CenterNet> create(
      const std::string &model_name, bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit CenterNet();
  CenterNet(const CenterNet &) = delete;

 public:
  virtual ~CenterNet();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function of get result of the CenterNet network.
   *
   * @param img Input data of input image (cv::Mat).
   *
   * @return CenterNetResult.
   *
   */
  virtual vitis::ai::CenterNetResult run(const cv::Mat &img) = 0;

  /**
   * @brief Function to get running results of the CenterNet network in
   * batch mode.
   *
   * @param imgs Input data of input images (vector<cv::Mat>).
   * The size of input images need equal to or less than
   * batch size obtained by get_input_batch.
   *
   * @return The vector of CenterNetResult.
   *
   */
  virtual std::vector<vitis::ai::CenterNetResult> run(
      const std::vector<cv::Mat> &imgs) = 0;

  /**
   * @brief Function to get InputWidth of the CenterNet network (input image columns).
   *
   * @return InputWidth of the CenterNet network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the CenterNet network (input image rows).
   *
   *@return InputHeight of the CenterNet network.
   */

  virtual int getInputHeight() const = 0;

  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be different. This depends on
   *the IP used.
   *
   *@return Batch size.
   */
  virtual size_t get_input_batch() const = 0;
};
}  // namespace ai
}  // namespace vitis
