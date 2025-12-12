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
 * Filename: Bosch_Ssdobjdet.hpp
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
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {

struct Bosch_SsdobjdetResult{
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// index num of the class.
  struct BoundingBox {
    /// Classification
    int label;
    /// Confidence
    float score;
    /// x-coordinate. x is normalized relative to the input image columns.
    /// Range from 0 to 1.
    float x;
    /// y-coordinate. y is normalized relative to the input image rows.
    /// Range from 0 to 1.
    float y;
    /// Width. Width is normalized relative to the input image columns,
    /// Range from 0 to 1.
    float width;
    /// Height. Heigth is normalized relative to the input image rows,
    /// Range from 0 to 1.
    float height;
  };
  /// All objects, a vector of BoundingBox
  std::vector<BoundingBox> bboxes;
};

/**
 * @brief Base class for Bosch_Ssdobjdet (production recognication)
 *
 * Input is an image (cv:Mat).
 *
 * Output is a struct of detected results, named Bosch_SsdobjdetResult.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_Bosch_Ssdobjdet.jpg");
   auto Bosch_Ssdobjdet = vitis::ai::Bosch_Ssdobjdet::create("bosch_fcnsemsegt",true);
   auto result = Bosch_Ssdobjdet->run(img);
   std::cout << result.width <<"\n";
   @endcode
 *
 */
class Bosch_Ssdobjdet {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Bosch_Ssdobjdet.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of Bosch_Ssdobjdet class.
   *
   */
  static std::unique_ptr<Bosch_Ssdobjdet> create(
      const std::string &model_name, bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Bosch_Ssdobjdet();
  Bosch_Ssdobjdet(const Bosch_Ssdobjdet &) = delete;

 public:
  virtual ~Bosch_Ssdobjdet();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function of get result of the Bosch_Ssdobjdet network.
   *
   * @param img Input data of input image (cv::Mat).
   *
   * @return Bosch_SsdobjdetResult.
   *
   */
  virtual vitis::ai::Bosch_SsdobjdetResult run(const cv::Mat &img) = 0;

  /**
   * @brief Function to get running results of the Bosch_Ssdobjdet network in
   * batch mode.
   *
   * @param imgs Input data of input images (vector<cv::Mat>).
   * The size of input images need equal to or less than
   * batch size obtained by get_input_batch.
   *
   * @return The vector of Bosch_SsdobjdetResult.
   *
   */
  virtual std::vector<vitis::ai::Bosch_SsdobjdetResult> run(
      const std::vector<cv::Mat> &imgs) = 0;

  /**
   * @brief Function to get InputWidth of the Bosch_Ssdobjdet network (input image columns).
   *
   * @return InputWidth of the Bosch_Ssdobjdet network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the Bosch_Ssdobjdet network (input image rows).
   *
   *@return InputHeight of the Bosch_Ssdobjdet network.
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
