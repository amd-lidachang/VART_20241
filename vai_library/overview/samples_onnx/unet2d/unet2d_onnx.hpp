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
#pragma once
#include <assert.h>
#include <glog/logging.h>
#include <algorithm>  // std::generate
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vector>

#include <vitis/ai/profiling.hpp>
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/onnx_task.hpp"

using namespace std;

struct Unet2DOnnxResult {
  std::vector<float> data;
};


static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

class Unet2DOnnx : public OnnxTask {
 public:
  static std::unique_ptr<Unet2DOnnx> create(const std::string& model_name) {
    return std::unique_ptr<Unet2DOnnx>(new Unet2DOnnx(model_name));
  }
  virtual ~Unet2DOnnx() {}

  Unet2DOnnx(const std::string& model_name) : OnnxTask(model_name) {}

  Unet2DOnnx(const Unet2DOnnx&) = delete;

  std::vector<Unet2DOnnxResult> run(const std::vector<float*> batch_images, int len) {
    // print name/shape of inputs
    std::vector<std::string> input_names = get_input_names();

    // print name/shape of outputs
    std::vector<std::string> output_names = get_output_names();

    std::vector<std::vector<int64_t>> input_shapes = get_input_shapes();
    std::vector<std::vector<int64_t>> output_shapes = get_output_shapes();

    // Assume model has 1 input node and 1 output node.
    assert(input_names.size() == 1 && output_names.size() == 1);

    // Create a single Ort tensor of random numbers
    auto input_shape = input_shapes[0];
    int total_number_elements = calculate_product(input_shape);
    std::vector<float> input_tensor_values(total_number_elements, 0.f);
    auto hw_batch = input_shape[0];
    auto valid_batch = std::min((int)hw_batch, (int)batch_images.size());
    __TIC__(PRE)
    this->preprocess(batch_images, len, input_tensor_values, input_shape,
                     valid_batch);
    __TOC__(PRE)

    std::vector<Ort::Value> input_tensors = convert_input(
        input_tensor_values, input_tensor_values.size(), input_shape);

    __TIC__(RUN)
    std::vector<Ort::Value> output_tensors;
    run_task(input_tensors, output_tensors);
    __TOC__(RUN)

    __TIC__(POST)
    auto results = this->postprocess(output_tensors[0], valid_batch);
    __TOC__(POST)
    return results;
  }

 protected:
  void preprocess(const std::vector<float*>& images, int len,
                  std::vector<float>& input_tensor_values,
                  std::vector<int64_t>& input_shape, int valid_batch);

  std::vector<Unet2DOnnxResult> postprocess(Ort::Value& output_tensor,
                                          int valid_batch);
};

std::vector<Unet2DOnnxResult> Unet2DOnnx::postprocess(Ort::Value& output_tensor,
                                        int valid_batch) {
  std::vector<Unet2DOnnxResult> results;
  auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
  // auto batch = output_shape[0];
  // auto channel = output_shape[1];
  auto width = output_shape[3];
  auto height = output_shape[2];
  // LOG(INFO) << "batch:" << batch << ", channel:" << channel
  //          << ", width:" << width << ", height:" << height;
  auto output_tensor_ptr = output_tensor.GetTensorMutableData<float>();
  for (auto index = 0; index < valid_batch; ++index) {
    float* src = output_tensor_ptr + index*(width*height);
    std::vector<float> v(width*height);  //  shili
    memcpy(v.data(), src, width*height*sizeof(float));
    results.emplace_back(Unet2DOnnxResult{v});
  }
  return results;
}

void Unet2DOnnx::preprocess(const std::vector<float*>& images,
                       int len,
                       std::vector<float>& input_tensor_values,
                       std::vector<int64_t>& input_shape, int valid_batch) {
  // auto batch = input_shape[0];
  auto channel = input_shape[1];
  auto height = input_shape[2];
  auto width = input_shape[3];
  auto batch_size = channel * height * width;
  int c_hw = height*width;

  int edge = (int)sqrt(len/channel);
  int dx = (edge - width)/2;
  for (auto index = 0; index < valid_batch; ++index) {
    float* p = input_tensor_values.data() + batch_size * index;
    // hwc -> chw
    for(int h=0; h<height; ++h ){
      for(int w=0; w<width; ++w ){
        float* src_base = images[index] + (edge*(dx+h) + dx+w)*channel;
        int c_1 = h * width + w;
        for(int c=0; c<(int)channel; ++c ){
           // p[ c * height * width + h * width + w] = *(src_base+c);
           p[ c * c_hw + c_1] = *(src_base+c);
        } // end for c
      }  // end for w
    } // end for h
  }// end for index
}

