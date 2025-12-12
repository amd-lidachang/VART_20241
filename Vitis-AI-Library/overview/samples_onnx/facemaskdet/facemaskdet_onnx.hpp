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
#include "vitis/ai/nnpp/apply_nms.hpp"

using namespace std;
using namespace cv;
using namespace vitis::ai;

// return value
struct OnnxFacemaskdetResult {
  int width;
  int height;
  struct BoundingBox {
    int label;
    /// Confidence. The value ranges from 0 to 1.
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
  /// All objects, The vector of BoundingBox.
  std::vector<BoundingBox> bboxes;
};

// model class
class OnnxFacemaskdet : public OnnxTask {
 public:
  static std::unique_ptr<OnnxFacemaskdet> create(const std::string& model_name) {
    return std::unique_ptr<OnnxFacemaskdet>(new OnnxFacemaskdet(model_name));
  }

 protected:
  explicit OnnxFacemaskdet(const std::string& model_name);
  OnnxFacemaskdet(const OnnxFacemaskdet&) = delete;

 public:
  virtual ~OnnxFacemaskdet() {}
  virtual std::vector<OnnxFacemaskdetResult> run(const std::vector<cv::Mat>& mats);

 private:
  std::vector<OnnxFacemaskdetResult> postprocess();
  OnnxFacemaskdetResult postprocess(int idx);
  void preprocess(const cv::Mat& image, int idx);
  void preprocess(const std::vector<cv::Mat>& mats);

 private:
  std::vector<float> input_tensor_values;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;

  int real_batch;
  int batch_size;
  std::vector<float*> input_tensor_ptr;
  std::vector<float*> output_tensor_ptr;
  int  channel =0;
  int  height  =0; 
  int  width   =0;

  float conf_thresh = 0.2f;
  float nms_thresh = 0.6f;
  int biases[12]= {12,18,37,49,  52,132,115,73,  119,199,242,238};
};

void OnnxFacemaskdet::preprocess(const cv::Mat& image, int idx) {
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(getInputWidth(), getInputHeight()));
  set_input_image_rgb(resized_image, input_tensor_values.data() + batch_size * idx, 
                  std::vector<float>{0,0,0}, std::vector<float>{0.00392157,0.00392157,0.00392157});
  return;
}

// preprocess
void OnnxFacemaskdet::preprocess(const std::vector<cv::Mat>& mats) {
  real_batch = std::min((int)input_shapes_[0][0], (int)mats.size());
  for (auto index = 0; index < real_batch; ++index) {
    preprocess(mats[index], index);
  }
}

inline float sigmoid(float src) { return (1.0f / (1.0f + exp(-src))); }

// postprocess
OnnxFacemaskdetResult OnnxFacemaskdet::postprocess(int idx) {
  vector<vector<float>> boxes;
  int num_classes = 2;
  int conf_box = 5+num_classes; // 7.   // 2 is classes num
  int anchor_cnt = 3;
  for(int i=0; i<2; i++) {  // 2 output layers  // 21x32x32       21x16x16 : 
     //  21 is 7x3,    3 is channel,  7 is fields in each channel
     auto conf_desigmoid = -logf(1.0f / conf_thresh - 1.0f);
     int ha= output_shapes_[i][2], wa= output_shapes_[i][3];
     int b1size = ha*wa*output_shapes_[i][1];
#define POS(C) ((C)*ha*wa+h*wa+w)
     for (int h = 0; h < ha; ++h) {
       for (int w = 0; w < wa; ++w) {
         for (int c = 0; c < anchor_cnt; ++c) {  // loop 3
             // int idx0 = ((h * width + w) * anchor_cnt + c) * conf_box;
             if (output_tensor_ptr[i][POS(c*conf_box+4) + idx*b1size] < conf_desigmoid) continue;
             vector<float> box;
     
             float obj_score =  sigmoid(output_tensor_ptr[i][POS(c*conf_box + 4) + idx*b1size] );
             box.push_back((w + sigmoid(output_tensor_ptr[i][POS(c*conf_box + 0) + idx*b1size] )) / width);
             box.push_back((h + sigmoid(output_tensor_ptr[i][POS(c*conf_box + 1) + idx*b1size] )) / height);
             box.push_back(exp(output_tensor_ptr[i][POS(c*conf_box + 2) + idx*b1size]) * biases[2 * c + 2 * anchor_cnt * i] / input_shapes_[0][3]);
             box.push_back(exp(output_tensor_ptr[i][POS(c*conf_box + 3) + idx*b1size]) * biases[2 * c + 2 * anchor_cnt * i + 1] / input_shapes_[0][2]);
             box.push_back(-1);
             box.push_back(obj_score);
             for (int p = 0; p < num_classes; p++) {
               box.push_back(obj_score * sigmoid(output_tensor_ptr[i][POS(c*conf_box + 5+p) +idx*b1size]));
             }
             boxes.push_back(box);
         }
       }
     }
  }
  /* Apply the computation for NMS */
  vector<vector<float>> res;
  vector<float> scores(boxes.size());
  for (int k = 0; k < num_classes; k++) {
    transform(boxes.begin(), boxes.end(), scores.begin(), [k](auto& box) {
      box[4] = k;
      return box[6 + k];
    });
    vector<size_t> result_k;
    applyNMS(boxes, scores, nms_thresh, conf_thresh, result_k, 1);
    transform(result_k.begin(), result_k.end(), back_inserter(res),
              [&boxes](auto& k) { return boxes[k]; });
  }

  OnnxFacemaskdetResult result{(int)input_shapes_[0][3], (int)input_shapes_[0][2]};
  for (size_t i = 0; i < res.size(); ++i) {
    if (res[i][res[i][4] + 6] > conf_thresh) {
      OnnxFacemaskdetResult::BoundingBox yolo_res;
      yolo_res.score = res[i][res[i][4] + 6];
      yolo_res.label = res[i][4];
      yolo_res.x = res[i][0] - res[i][2] / 2.0;
      yolo_res.y = res[i][1] - res[i][3] / 2.0;
      yolo_res.width = res[i][2];
      yolo_res.height = res[i][3];
      result.bboxes.push_back(yolo_res);
    }
  }
  return result;
}

std::vector<OnnxFacemaskdetResult> OnnxFacemaskdet::postprocess() {
  std::vector<OnnxFacemaskdetResult> ret;
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

OnnxFacemaskdet::OnnxFacemaskdet(const std::string& model_name) : OnnxTask(model_name) {
  int total_number_elements = calculate_product(input_shapes_[0]);
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  channel = input_shapes_[0][1];
  height = input_shapes_[0][2];
  width = input_shapes_[0][3];
  batch_size = channel * height * width;
  input_tensor_ptr.resize(1);
  output_tensor_ptr.resize(2);
}

std::vector<OnnxFacemaskdetResult> OnnxFacemaskdet::run(const std::vector<cv::Mat>& mats) {
  __TIC__(total)
  __TIC__(preprocess)

  if (input_tensors.size()) {
    input_tensors[0] = Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0]);
  } else {
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0]));
  }
  preprocess(mats);

  __TOC__(preprocess)

  __TIC__(session_run)
  run_task(input_tensors, output_tensors);
  output_tensor_ptr[0] = output_tensors[0].GetTensorMutableData<float>();
  output_tensor_ptr[1] = output_tensors[1].GetTensorMutableData<float>();
  __TOC__(session_run)

  __TIC__(postprocess)
  std::vector<OnnxFacemaskdetResult> ret = postprocess();
  __TOC__(postprocess)
  __TOC__(total)
  return ret;
}



