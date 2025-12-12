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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "vitis/ai/onnx_task.hpp"
#include "vitis/ai/profiling.hpp"

DEF_ENV_PARAM(ENABLE_YOLO_DEBUG, "0");

using namespace std;
using namespace cv;

static float overlap(float x1, float w1, float x2, float w2) {
  float left = max(x1 - w1 / 2.0, x2 - w2 / 2.0);
  float right = min(x1 + w1 / 2.0, x2 + w2 / 2.0);
  return right - left;
}

static float cal_iou(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0 / union_area;
}

static void applyNMS(const vector<vector<float>>& boxes,
                     const vector<float>& scores, const float nms,
                     const float conf, vector<size_t>& res) {
  const size_t count = boxes.size();
  vector<pair<float, size_t>> order;
  for (size_t i = 0; i < count; ++i) {
    order.push_back({scores[i], i});
  }
  stable_sort(order.begin(), order.end(),
              [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
                return ls.first > rs.first;
              });
  vector<size_t> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
            [](auto& km) { return km.second; });
  vector<bool> exist_box(count, true);

  for (size_t _i = 0; _i < count; ++_i) {
    size_t i = ordered[_i];
    if (!exist_box[i]) continue;
    if (scores[i] < conf) {
      exist_box[i] = false;
      continue;
    }
    /* add a box as result */
    res.push_back(i);
    // cout << "nms push "<< i<<endl;
    for (size_t _j = _i + 1; _j < count; ++_j) {
      size_t j = ordered[_j];
      if (!exist_box[j]) continue;
      float ovr = 0.0;
      ovr = cal_iou(boxes[j], boxes[i]);
      if (ovr >= nms) exist_box[j] = false;
    }
  }
}

static cv::Mat letterbox(const cv::Mat& im, int w, int h) {
  float scale = min((float)w / (float)im.cols, (float)h / (float)im.rows);
  int new_w = im.cols * scale;
  int new_h = im.rows * scale;
  cv::Mat img_res;
  cv::resize(im, img_res, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  cv::Mat new_img(cv::Size(w, h), CV_8UC3, cv::Scalar(114, 114, 114));
  int x = (w - new_w) / 2;
  int y = (h - new_h) / 2;
  auto rect = cv::Rect{x, y, new_w, new_h};
  img_res.copyTo(new_img(rect));
  return new_img;
}

static void correct_region_boxes(vector<vector<float>>& boxes, int n, int w,
                                 int h, int netw, int neth) {
  int new_w = 0;
  int new_h = 0;

  if (((float)netw / w) < ((float)neth / h)) {
    new_w = netw;
    new_h = (h * netw) / w;
  } else {
    new_h = neth;
    new_w = (w * neth) / h;
  }
  for (int i = 0; i < n; ++i) {
    boxes[i][0] = (boxes[i][0] - (netw - new_w) / 2. / netw) /
                  ((float)new_w / (float)netw);
    boxes[i][1] = (boxes[i][1] - (neth - new_h) / 2. / neth) /
                  ((float)new_h / (float)neth);
    boxes[i][2] *= (float)netw / new_w;
    boxes[i][3] *= (float)neth / new_h;
  }
}

// return value
struct Yolov5OnnxResult {
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
class Yolov5Onnx : public OnnxTask {
 public:
  static std::unique_ptr<Yolov5Onnx> create(const std::string& model_name,
                                            const float conf_thresh_) {
    return std::unique_ptr<Yolov5Onnx>(
        new Yolov5Onnx(model_name, conf_thresh_));
  }

 protected:
  explicit Yolov5Onnx(const std::string& model_name, const float conf_thresh_);
  Yolov5Onnx(const Yolov5Onnx&) = delete;

 public:
  virtual ~Yolov5Onnx() {}
  virtual std::vector<Yolov5OnnxResult> run(const std::vector<cv::Mat>& mats);
  virtual Yolov5OnnxResult run(const cv::Mat& mats);

 private:
  std::vector<Yolov5OnnxResult> postprocess(const std::vector<int>& ws,
                                            const std::vector<int>& hs);
  Yolov5OnnxResult postprocess(int idx, int width, int height);
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
  int output_tensor_size = 4;
  int channel = 0;
  int sHeight = 0;
  int sWidth = 0;
  float conf_thresh = 0.f;
  float nms_thresh = 0.65f;
  int biases[18] = {10, 13, 16,  30,  33, 23,  30,  61,  62,
                    45, 59, 119, 116, 90, 156, 198, 373, 326};
  int num_classes = 80;
  int anchor_cnt = 3;
};

void Yolov5Onnx::preprocess(const cv::Mat& image, int idx) {
  cv::Mat resized_image;
  resized_image = letterbox(image, sWidth, sHeight);
  set_input_image_rgb(resized_image,
                      input_tensor_values.data() + batch_size * idx,
                      std::vector<float>{0, 0, 0},
                      std::vector<float>{0.00392156, 0.00392156, 0.00392156});
  return;
}
// preprocess
void Yolov5Onnx::preprocess(const std::vector<cv::Mat>& mats) {
  real_batch = std::min((int)input_shapes_[0][0], (int)mats.size());
  for (auto index = 0; index < real_batch; ++index) {
    preprocess(mats[index], index);
  }
}

inline float sigmoid(float src) { return (1.0f / (1.0f + exp(-src))); }

// postprocess
Yolov5OnnxResult Yolov5Onnx::postprocess(int idx, int width, int height) {
  vector<vector<float>> boxes;

  int conf_box = 5 + num_classes;

  auto conf_desigmoid = -logf(1.0f / conf_thresh - 1.0f);

  for (int i = 1; i < output_tensor_size; i++) {
    int ch = output_shapes_[i][1];
    int ha = output_shapes_[i][2];
    int wa = output_shapes_[i][3];
    int outputBatch = ch * ha * wa * output_shapes_[i][4];
    if (ENV_PARAM(ENABLE_YOLO_DEBUG)) {
      LOG(INFO) << "channel=" << ch << ", height=" << ha << ", width=" << wa
                << endl;
    }
    boxes.reserve(boxes.size() + ch * ha * wa);

#define POS(C) ((C)*conf_box * wa * ha + h * wa * conf_box + w * conf_box)
    for (int h = 0; h < ha; ++h) {
      for (int w = 0; w < wa; ++w) {
        for (int c = 0; c < anchor_cnt; ++c) {  // loop 3
          float score = output_tensor_ptr[i][POS(c) + 4 + idx * outputBatch];
          if (score <= conf_desigmoid) continue;
          vector<float> box;

          float obj_score = sigmoid(score);
          vector<float> out(4);
          for (int index = 0; index < 4; index++) {
            out[index] =
                output_tensor_ptr[i][POS(c) + index + idx * outputBatch];
          }
          box.push_back((w - 0.5 + 2 * sigmoid(out[0])) / wa);
          box.push_back((h - 0.5 + 2 * sigmoid(out[1])) / ha);
          box.push_back(pow(2 * sigmoid(out[2]), 2) *
                        biases[2 * c + 2 * anchor_cnt * (i - 1) + 0] /
                        (float)(sWidth));
          box.push_back(pow(2 * sigmoid(out[3]), 2) *
                        biases[2 * c + 2 * anchor_cnt * (i - 1) + 1] /
                        (float)(sHeight));
          box.push_back(-1);
          box.push_back(obj_score);
          for (int p = 5; p < conf_box; p++) {
            box.push_back(
                obj_score *
                sigmoid(output_tensor_ptr[i][POS(c) + p + idx * outputBatch]));
          }
          boxes.push_back(box);
        }
      }
    }
  }

  correct_region_boxes(boxes, boxes.size(), width, height, sWidth, sHeight);

  /* Apply the computation for NMS */
  vector<vector<float>> res;
  vector<float> scores(boxes.size());
  for (int k = 0; k < num_classes; k++) {
    transform(boxes.begin(), boxes.end(), scores.begin(), [k](auto& box) {
      box[4] = k;
      return box[6 + k];
    });
    vector<size_t> result_k;
    applyNMS(boxes, scores, nms_thresh, conf_thresh, result_k);
    transform(result_k.begin(), result_k.end(), back_inserter(res),
              [&boxes](auto& k) { return boxes[k]; });
  }
  // cout<<"res size:"<<res.size()<<std::endl;
  vector<Yolov5OnnxResult::BoundingBox> results;
  for (const auto& r : res) {
    auto score = r[r[4] + 6];
    if (score > conf_thresh) {
      Yolov5OnnxResult::BoundingBox result;
      result.score = score;
      result.label = r[4];

      result.x = (r[0] - r[2] / 2.0f);
      result.y = (r[1] - r[3] / 2.0f);
      result.width = r[2];
      result.height = r[3];
      results.push_back(result);
    }
  }
  return Yolov5OnnxResult{results};
}

std::vector<Yolov5OnnxResult> Yolov5Onnx::postprocess(
    const std::vector<int>& ws, const std::vector<int>& hs) {
  std::vector<Yolov5OnnxResult> ret;
  for (auto index = 0; index < (int)real_batch; ++index) {
    ret.emplace_back(postprocess(index, ws[index], hs[index]));
  }
  return ret;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

Yolov5Onnx::Yolov5Onnx(const std::string& model_name, const float conf_thresh_)
    : OnnxTask(model_name) {
  int total_number_elements = calculate_product(input_shapes_[0]);
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  channel = input_shapes_[0][1];
  sHeight = input_shapes_[0][2];
  sWidth = input_shapes_[0][3];
  batch_size = channel * sHeight * sWidth;
  input_tensor_ptr.resize(1);
  output_tensor_ptr.resize(output_tensor_size);
  conf_thresh = conf_thresh_;
}

Yolov5OnnxResult Yolov5Onnx::run(const cv::Mat& mats) {
  return run(vector<cv::Mat>(1, mats))[0];
}

std::vector<Yolov5OnnxResult> Yolov5Onnx::run(
    const std::vector<cv::Mat>& mats) {
  __TIC__(total)
  __TIC__(preprocess)
  preprocess(mats);
  // std::cout<<"pre process is done"<<std::endl;
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
  for (int i = 1; i < output_tensor_size; i++) {
    output_tensor_ptr[i] = output_tensors[i].GetTensorMutableData<float>();
  }
  __TOC__(session_run)
  vector<int> ws, hs;
  for (auto input_image : mats) {
    ws.push_back(input_image.cols);
    hs.push_back(input_image.rows);
  }
  __TIC__(postprocess)
  std::vector<Yolov5OnnxResult> ret = postprocess(ws, hs);
  __TOC__(postprocess)
  __TOC__(total)
  return ret;
}
