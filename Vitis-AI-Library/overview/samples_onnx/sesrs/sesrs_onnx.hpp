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

#include <opencv2/imgproc/imgproc_c.h>
#include <algorithm>  // std::generate
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vector>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/onnx_task.hpp"
#include "vitis/ai/profiling.hpp"

using namespace std;
using namespace cv;
using namespace vitis::ai;

// return value
struct OnnxSesrsResult {
  int width;
  int height;
  cv::Mat mat;
};

// model class
class OnnxSesrs : public OnnxTask {
 public:
  static std::unique_ptr<OnnxSesrs> create(const std::string& model_name) {
    return std::unique_ptr<OnnxSesrs>(new OnnxSesrs(model_name));
  }

 protected:
  explicit OnnxSesrs(const std::string& model_name);
  OnnxSesrs(const OnnxSesrs&) = delete;

 public:
  virtual ~OnnxSesrs() {}
  virtual std::vector<OnnxSesrsResult> run(const std::vector<cv::Mat>& mats);

 private:
  std::vector<OnnxSesrsResult> postprocess();
  OnnxSesrsResult postprocess(int idx);
  cv::Mat set_output_image(float* data);
  void preprocess(const cv::Mat& image, int idx);
  void preprocess(const std::vector<cv::Mat>& mats);

 private:
  std::vector<float> input_tensor_values;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;

  int real_batch;
  int batch_size;
  std::vector<float*> output_tensor_ptr;
};

void OnnxSesrs::preprocess(const cv::Mat& image, int idx) {
  cv::Mat resized_image;
  if (image.cols != (int)getInputWidth() || image.rows != (int)getInputHeight()) {
     std::cerr <<"image size doesn't match the model\n"; 
     exit(0);
  }
  set_input_image_bgr(image, input_tensor_values.data() + batch_size * idx,
                  std::vector<float>{0.0f, 0.0f, 0.0f},
                  std::vector<float>{1.0f, 1.0f, 1.0f}
                 );
  return;
}

// preprocess
void OnnxSesrs::preprocess(const std::vector<cv::Mat>& mats) {
  real_batch = std::min((int)input_shapes_[0][0], (int)mats.size());
  for (auto index = 0; index < real_batch; ++index) {
    preprocess(mats[index], index);
  }
}

cv::Mat OnnxSesrs::set_output_image(float* data) {
  cv::Mat image(output_shapes_[0][2], output_shapes_[0][3], CV_8UC3 );
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < image.rows; h++) {
      for (int w = 0; w < image.cols; w++) {
        float image_data = data[c * image.rows * image.cols + h * image.cols + w];
        image.at<cv::Vec3b>(h, w)[c] = int8_t(image_data);
      }
    }
  }
  return image;
}

// postprocess
OnnxSesrsResult OnnxSesrs::postprocess(int idx) {
  float* fp = output_tensor_ptr[0] + idx * output_shapes_[0][1]*output_shapes_[0][2]*output_shapes_[0][3];
  cv::Mat mat = set_output_image(fp);
  OnnxSesrsResult result{ (int)getInputWidth(), (int)getInputHeight(), mat};
  return result;
}

std::vector<OnnxSesrsResult> OnnxSesrs::postprocess() {
  std::vector<OnnxSesrsResult> ret;
  for (auto index = 0; index < (int)real_batch; ++index) {
    ret.emplace_back(postprocess(index));
  }
  return ret;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

OnnxSesrs::OnnxSesrs(const std::string& model_name) : OnnxTask(model_name) {
  auto input_shape = input_shapes_[0];
  int total_number_elements = calculate_product(input_shape);
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  auto channel = input_shapes_[0][1];
  auto height = input_shapes_[0][2];
  auto width = input_shapes_[0][3];
  batch_size = channel * height * width;

  output_tensor_ptr.resize(1);
}

std::vector<OnnxSesrsResult> OnnxSesrs::run(const std::vector<cv::Mat>& mats) {
  __TIC__(total)
  __TIC__(preprocess)
  preprocess(mats);

  if (input_tensors.size()) {
    input_tensors[0] = Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0]);
  } else {
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0]));
  }
  __TOC__(preprocess)

  __TIC__(session_run)
  run_task(input_tensors, output_tensors);
  output_tensor_ptr[0] = output_tensors[0].GetTensorMutableData<float>();
  __TOC__(session_run)

  __TIC__(postprocess)
  std::vector<OnnxSesrsResult> ret = postprocess();
  __TOC__(postprocess)
  __TOC__(total)
  return ret;
}

