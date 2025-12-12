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
 * Filename: x_autonomous3d.hpp
 *
 * Description:
 * This network is used to detecting objects from a input points data.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/nnpp/x_autonomous3d.hpp>

namespace vitis {
namespace ai {

// using X_Autonomous3DResult = CenterPointResult;

class X_Autonomous3D {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * X_Autonomous3D
   *
   * @param model_name_0 Model 0 name
   *
   * @param model_name_1 Model 1 name
   *
   * @return An instance of X_Autonomous3D class.
   */
  static std::unique_ptr<X_Autonomous3D> create(
      const std::string& model_name_0, const std::string& model_name_1);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit X_Autonomous3D();
  X_Autonomous3D(const X_Autonomous3D&) = delete;
  X_Autonomous3D& operator=(const X_Autonomous3D&) = delete;

 public:
  virtual ~X_Autonomous3D();
  /**
   * @endcond
   */
  /**
   * @brief Function to get InputWidth of the X_Autonomous3D network (input
   * image columns).
   *
   * @return InputWidth of the X_Autonomous3D network
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeight of the X_Autonomous3D network (input
   *image rows).
   *
   *@return InputHeight of the X_Autonomous3D network.
   */
  virtual int getInputHeight() const = 0;

  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be different. This depends on
   *the IP used.
   *
   * @return Batch size.
   */
  virtual size_t get_input_batch() const = 0;

  /**
   * @brief Function to get running results of the X_Autonomous3D network.
   *
   * @param input 3D lidar data
   *
   * @return X_Autonomous3DResult.
   *
   */
  virtual vitis::ai::X_Autonomous3DResult run(
      const std::vector<float>& input) = 0;

  /**
   * @brief Function to get running results of the X_Autonomous3D network in
   * batch mode.
   *
   * @param inputs Vector of 3D lidar data. The size of vector should equal to
   * batch size.
   *
   * @return The vector of X_Autonomous3DResult.
   *
   */

  virtual std::vector<X_Autonomous3DResult> run(
      const std::vector<std::vector<float>>& inputs) = 0;
};

}  // namespace ai
}  // namespace vitis
