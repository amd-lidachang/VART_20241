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
#include "./bts_imp.hpp"
#include <cmath>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

DEF_ENV_PARAM(ENABLE_BTS_DEBUG, "0");
DEF_ENV_PARAM(ENABLE_BTS_SET_INPUT_DEBUG, "0");

void sigmoid_int8_n(int8_t* input, float scale, int size, float* output) {
  for (int i = 0; i < size; ++i) {
    output[i] = (1. / (1. + std::exp(-1. * (int)(input[i]) * scale)));
  }
}

void hard_sigmoid_int8_n(int8_t* input, float scale, int size, float* output) {
  for (int i = 0; i < size; ++i) {
    float val = input[i] * scale;
    if (val <= -3.f) {
      output[i] = 0.f;
    } else if (val >= 3.f) {
      output[i] = 1.f;
    } else {
      output[i] = val / 6.f + 0.5;
    }
    // if (i < 10) {
    //  std::cout << "before h-sigmoid:" << (int)input[i] << ", "
    //            << "val :" << val << ", "
    //            << "after h-sigmoid:" << output[i] << std::endl;
    //}
  }
}

void hard_sigmoid_int8_n(int8_t* last_layer, float last_layer_scale, int size,
                         int8_t* output, float output_fix_scale) {
  for (int i = 0; i < size; ++i) {
    float val = last_layer[i] * last_layer_scale;
    if (val <= -3.f) {
      val = 0.f;
    } else if (val >= 3.f) {
      val = 1.f;
    } else {
      val = val / 6.f + 0.5;
    }

    int candidate = std::round(val * output_fix_scale);
    if (candidate < -128) {
      candidate = -128;
    } else if (candidate > 127) {
      candidate = 127;
    }
    output[i] = candidate;
  }
}

void multiple_int8_n(int8_t* last_layer, float last_layer_scale, int size,
                     uint16_t* output, float output_fix_scale, float mul_factor,
                     float depth_scaled_factor) {
  for (int i = 0; i < size; ++i) {
    int val = std::round(last_layer[i] * last_layer_scale * mul_factor *
                         output_fix_scale);
    if (val > 127) {
      val = 127;
    } else if (val < -128) {
      val = -128;
    }
    output[i] = val / output_fix_scale * depth_scaled_factor;
  }
}

BTSImp::BTSImp(const std::string& model_name, bool need_preprocess)
    : BTS(model_name, need_preprocess) {}

BTSImp::~BTSImp() {}

void BTSImp::run_internal(const cv::Mat& input_image) {
  __TIC__(BTS_IMG_RESIZE)
  cv::Mat image;
  int width = getInputWidth();
  int height = getInputHeight();
  auto size = cv::Size(width, height);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0, 0, cv::INTER_LINEAR);
  } else {
    image = input_image;
  }
  __TOC__(BTS_IMG_RESIZE)
  __TIC__(BTS_SET_IMG)
  // auto input_tensors = configurable_dpu_task_->getInputTensor();
  // auto input_scale = vitis::ai::library::tensor_scale(input_tensors[0][0]);
  // auto input_ptr = (int8_t*)input_tensors[0][0].get_data(0);
  // LOG_IF(INFO, ENV_PARAM(ENABLE_BTS_DEBUG)) << "input scale:" << input_scale;
  configurable_dpu_task_->setInputImageRGB(image);
  __TOC__(BTS_SET_IMG)

  __TIC__(BTS_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(BTS_DPU)
}

void BTSImp::run_internal(const std::vector<cv::Mat>& input_images) {
  __TIC__(BTS_IMG_RESIZE_BATCH)
  std::vector<cv::Mat> images;
  int width = getInputWidth();
  int height = getInputHeight();
  auto size = cv::Size(width, height);
  for (auto i = 0u; i < input_images.size(); i++) {
    if (size != input_images[i].size()) {
      cv::Mat img;
      cv::resize(input_images[i], img, size, 0, 0, cv::INTER_LINEAR);
      images.push_back(img);
    } else {
      images.push_back(input_images[i]);
    }
  }
  __TOC__(BTS_IMG_RESIZE_BATCH)
  __TIC__(BTS_SET_IMG_BATCH)
  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(BTS_SET_IMG_BATCH)

  __TIC__(BTS_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(BTS_DPU)
}

BTSResult BTSImp::bts_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, size_t batch_idx) {
  __TIC__(BTS_POST_NEW)
  BTSResult result;
  auto input_scale = vitis::ai::library::tensor_scale(input_tensors[0][0]);
  auto output_scale = vitis::ai::library::tensor_scale(output_tensors[0][0]);
  LOG_IF(INFO, ENV_PARAM(ENABLE_BTS_DEBUG)) << "input_scale:" << input_scale;
  LOG_IF(INFO, ENV_PARAM(ENABLE_BTS_DEBUG)) << "output_scale:" << output_scale;
  auto input_width = getInputWidth();
  auto input_height = getInputHeight();
  LOG_IF(INFO, ENV_PARAM(ENABLE_BTS_DEBUG))
      << "input_width:" << input_width << ", "
      << "input_height:" << input_height;
  float mul_factor = 10.0f;
  float depth_scaled_factor = 1000.0f;
  auto output_tensor_name = output_tensors[0][0].name;
  auto output_width = output_tensors[0][0].width;
  auto output_height = output_tensors[0][0].height;
  // vector<float> real_result(output_width * output_height);
  vector<uint16_t> bitmap(output_width * output_height);
  LOG_IF(INFO, ENV_PARAM(ENABLE_BTS_DEBUG))
      << "output_tensor_name:" << output_tensor_name;
  // float hard_sigmoid_output_scale = 0.0078125;
  // float multiple_output_scale = 0.125;
  if (output_tensor_name.find("Hardsigmoid") == std::string::npos) {
    vector<int8_t> hard_sigmoid_output(bitmap.size());
    __TIC__(BTS_HARD_SIGMOID)
    hard_sigmoid_int8_n((int8_t*)output_tensors[0][0].get_data(batch_idx),
                        output_scale, hard_sigmoid_output.size(),
                        hard_sigmoid_output.data(), 128);
    __TOC__(BTS_HARD_SIGMOID)
    __TIC__(BTS_MUL)
    multiple_int8_n(hard_sigmoid_output.data(), 1.f / 128, bitmap.size(),
                    bitmap.data(), 8, mul_factor, depth_scaled_factor);
    __TOC__(BTS_MUL)
  } else {
    __TIC__(BTS_MUL)
    multiple_int8_n((int8_t*)output_tensors[0][0].get_data(batch_idx),
                    output_scale, bitmap.size(), bitmap.data(), 8, mul_factor,
                    depth_scaled_factor);
    __TOC__(BTS_MUL)
  }

  // for (auto i = 0u; i < real_result.size(); ++i) {
  //  bitmap[i] = real_result[i] * depth_scaled_factor;
  //}
  result.width = input_width;
  result.height = input_height;
  result.depth_mat =
      cv::Mat(output_height, output_width, CV_16U, bitmap.data()).clone();
  // result.depth = real_result;

  __TOC__(BTS_POST_NEW)
  return result;
}

std::vector<BTSResult> BTSImp::bts_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config) {
  auto batch_size = input_tensors[0][0].batch;
  auto ret = std::vector<BTSResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.emplace_back(
        bts_post_process(input_tensors, output_tensors, config, i));
  }
  return ret;
}

BTSResult BTSImp::run(const cv::Mat& input_image) {
  run_internal(input_image);
  __TIC__(BTS_POST_ARM)
  // auto result = vitis::ai::bts_post_process(
  auto result = bts_post_process(configurable_dpu_task_->getInputTensor(),
                                 configurable_dpu_task_->getOutputTensor(),
                                 configurable_dpu_task_->getConfig(), 0);
  __TOC__(BTS_POST_ARM)
  // return std::move(result[0]);
  return result;
}

std::vector<BTSResult> BTSImp::run(const vector<cv::Mat>& input_images) {
  run_internal(input_images);
  __TIC__(BTS_POST_ARM)
  auto ret = bts_post_process(configurable_dpu_task_->getInputTensor(),
                              configurable_dpu_task_->getOutputTensor(),
                              configurable_dpu_task_->getConfig());
  __TOC__(BTS_POST_ARM)

  return ret;
}

}  // namespace ai
}  // namespace vitis
