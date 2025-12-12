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

static float sigmoid(float p) { return 1.0 / (1 + exp(-p * 1.0)); }

static void correct_region_boxes(vector<vector<float>>& boxes, int w, int h,
                                 int netw, int neth) {
  float scale = min((float)netw / (float)w, (float)neth / (float)h);

  auto dw = float(netw - round(w * scale)) / 2.0f;
  auto dh = float(neth - round(h * scale)) / 2.0f;

  for (auto& box : boxes) {
    box[0] = (box[0] - dw) / scale / w * netw;
    box[1] = (box[1] - dh) / scale / h * neth;
    box[2] = box[2] / scale / w * netw;
    box[3] = box[3] / scale / h * neth;
  }
}

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

static void letterbox(const cv::Mat& im, const cv::Mat& om, int w, int h) {
  float scale = min((float)w / (float)im.cols, (float)h / (float)im.rows);

  int new_w = round(im.cols * scale);
  int new_h = round(im.rows * scale);

  Mat img_res;
  if (im.size() != Size(new_w, new_h)) {
    cv::resize(im, img_res, Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  } else {
    img_res = im;
  }
  auto dw = float(w - new_w) / 2.0f;
  auto dh = float(h - new_h) / 2.0f;

  copyMakeBorder(img_res, om, int(round(dh - 0.1)), int(round(dh + 0.1)),
                 int(round(dw - 0.1)), int(round(dw + 0.1)),
                 cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

  return;
}

// return value
struct OfaYoloOnnxResult {
  /**
   *@struct BoundingBox
   *@brief Struct of detection result with an object.
   */
  struct BoundingBox {
    /// Classification.
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
class OfaYoloOnnx : public OnnxTask {
 public:
  static std::unique_ptr<OfaYoloOnnx> create(const std::string& model_name,
                                             const float conf_thresh_) {
    return std::unique_ptr<OfaYoloOnnx>(
        new OfaYoloOnnx(model_name, conf_thresh_));
  }

 protected:
  explicit OfaYoloOnnx(const std::string& model_name, const float conf_thresh_);
  OfaYoloOnnx(const OfaYoloOnnx&) = delete;

 public:
  virtual ~OfaYoloOnnx() {}
  virtual std::vector<OfaYoloOnnxResult> run(const std::vector<cv::Mat>& mats);
  virtual OfaYoloOnnxResult run(const cv::Mat& mats);

 private:
  std::vector<OfaYoloOnnxResult> postprocess();
  OfaYoloOnnxResult postprocess(int idx);
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
  float stride[4] = {0, 8, 16, 32};
  float conf_thresh = 0.f;
  float conf_desigmoid = 0.f;
  float nms_thresh = 0.65f;
  int num_classes = 80;
  int max_nms_num = 300;
  int max_boxes_num = 30000;
  int anchor_cnt = 3;
  int biases[18] = {10, 13, 16,  30,  33, 23,  30,  61,  62,
                    45, 59, 119, 116, 90, 156, 198, 373, 326};

  vector<int> ws;
  vector<int> hs;
};

void OfaYoloOnnx::preprocess(const cv::Mat& image, int idx) {
  cv::Mat resized_image(Size(sWidth, sHeight), CV_8UC3,
                        cv::Scalar(128, 128, 128));
  letterbox(image, resized_image, sWidth, sHeight);
  set_input_image_rgb(resized_image,
                      input_tensor_values.data() + batch_size * idx,
                      std::vector<float>{0, 0, 0},
                      std::vector<float>{0.00392156, 0.00392156, 0.00392156});
  return;
}

// preprocess
void OfaYoloOnnx::preprocess(const std::vector<cv::Mat>& mats) {
  real_batch = std::min((int)input_shapes_[0][0], (int)mats.size());
  ws.resize(real_batch);
  hs.resize(real_batch);

  for (auto i = 0; i < real_batch; ++i) {
    preprocess(mats[i], i);
    ws[i] = mats[i].cols;
    hs[i] = mats[i].rows;
  }
  return;
}

// postprocess
OfaYoloOnnxResult OfaYoloOnnx::postprocess(int idx) {
  vector<vector<float>> boxes;
  int conf_box = 5 + num_classes;
  for (int i = 1; i < output_tensor_size; i++) {
    int ch = output_shapes_[i][1];
    int ha = output_shapes_[i][2];
    int wa = output_shapes_[i][3];
    int outputBatch = ch * ha * wa * output_shapes_[i][4];
    if (ENV_PARAM(ENABLE_YOLO_DEBUG)) {
      LOG(INFO) << "channel=" << ch << ", height=" << ha << ", width=" << wa
                << ", stride=" << stride[i]
                << ", box_num=" << output_shapes_[i][4] << endl;
    }
    boxes.reserve(boxes.size() + ch * ha * wa);

#define POS(C) ((C)*ha * wa * conf_box + h * wa * conf_box + w * conf_box)
    for (int h = 0; h < ha; ++h) {
      for (int w = 0; w < wa; ++w) {
        for (int c = 0; c < anchor_cnt; ++c) {
          float score = output_tensor_ptr[i][POS(c) + 4 + idx * outputBatch];
          if (score < conf_desigmoid) continue;

          vector<float> box(6);
          vector<float> out(4);
          for (int index = 0; index < 4; index++) {
            out[index] =
                output_tensor_ptr[i][POS(c) + index + idx * outputBatch];
          }
          box[0] = (w + sigmoid(out[0]) * 2.0f - 0.5f) * stride[i];
          box[1] = (h + sigmoid(out[1]) * 2.0f - 0.5f) * stride[i];
          box[2] = pow(sigmoid(out[2]) * 2, 2) *
                   biases[2 * c + 2 * anchor_cnt * (i - 1)];
          box[3] = pow(sigmoid(out[3]) * 2, 2) *
                   biases[2 * c + 2 * anchor_cnt * (i - 1) + 1];
          float obj_score = sigmoid(score);
          auto conf_class_desigmoid = -logf(obj_score / conf_thresh - 1.0f);
          for (int p = 0; p < num_classes; p++) {
            float cls_score =
                output_tensor_ptr[i][POS(c) + 5 + p + idx * outputBatch];
            if (cls_score < conf_class_desigmoid) continue;
            box[4] = p;
            box[5] = obj_score * sigmoid(cls_score);
            boxes.push_back(box);
          }
        }
      }
    }
  }
  /* Apply the computation for NMS */
  if (static_cast<int>(boxes.size()) > max_boxes_num) {
    stable_sort(boxes.begin(), boxes.end(),
                [](vector<float> l, vector<float> r) { return l[5] > r[5]; });
    boxes.resize(max_boxes_num);
  }
  vector<vector<vector<float>>> boxes_for_nms(num_classes);
  vector<vector<float>> scores(num_classes);

  for (const auto& box : boxes) {
    boxes_for_nms[box[4]].push_back(box);
    scores[box[4]].push_back(box[5]);
  }

  vector<vector<float>> res;
  for (auto i = 0; i < num_classes; i++) {
    vector<size_t> result_k;
    applyNMS(boxes_for_nms[i], scores[i], nms_thresh, conf_thresh, result_k);
    res.reserve(res.size() + result_k.size());
    transform(result_k.begin(), result_k.end(), back_inserter(res),
              [&](auto& k) { return boxes_for_nms[i][k]; });
  }

  stable_sort(res.begin(), res.end(),
              [](vector<float> l, vector<float> r) { return l[5] > r[5]; });
  if (static_cast<int>(res.size()) > max_nms_num) {
    res.resize(max_nms_num);
  }

  /* Restore the correct coordinate frame of the original image */
  correct_region_boxes(res, ws[idx], hs[idx], sWidth, sHeight);

  vector<OfaYoloOnnxResult::BoundingBox> results;
  for (const auto& r : res) {
    if (r[5] > conf_thresh) {
      OfaYoloOnnxResult::BoundingBox result;
      result.score = r[5];
      result.label = r[4];

      result.x = (r[0] - r[2] / 2.0f) / sWidth;
      result.y = (r[1] - r[3] / 2.0f) / sHeight;
      result.width = r[2] / sWidth;
      result.height = r[3] / sHeight;
      results.push_back(result);
    }
  }
  return OfaYoloOnnxResult{results};
}

std::vector<OfaYoloOnnxResult> OfaYoloOnnx::postprocess() {
  std::vector<OfaYoloOnnxResult> ret;
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

OfaYoloOnnx::OfaYoloOnnx(const std::string& model_name,
                         const float conf_thresh_)
    : OnnxTask(model_name) {
  int total_number_elements = calculate_product(input_shapes_[0]);
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  channel = input_shapes_[0][1];
  sHeight = input_shapes_[0][2];
  sWidth = input_shapes_[0][3];
  batch_size = channel * sHeight * sWidth * input_shapes_[0][4];
  input_tensor_ptr.resize(1);
  output_tensor_ptr.resize(output_tensor_size);
  conf_thresh = conf_thresh_;
  conf_desigmoid = -logf(1.0f / conf_thresh - 1.0f);
  if (ENV_PARAM(ENABLE_YOLO_DEBUG)) {
    LOG(INFO) << "channel=" << channel << ", height=" << sHeight
              << ", width=" << sWidth << ", conf=" << conf_thresh << endl;
  }
}

OfaYoloOnnxResult OfaYoloOnnx::run(const cv::Mat& mats) {
  return run(vector<cv::Mat>(1, mats))[0];
}

std::vector<OfaYoloOnnxResult> OfaYoloOnnx::run(
    const std::vector<cv::Mat>& mats) {
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
  for (int i = 1; i < output_tensor_size; i++) {
    output_tensor_ptr[i] = output_tensors[i].GetTensorMutableData<float>();
  }
  __TOC__(session_run)

  __TIC__(postprocess)
  std::vector<OfaYoloOnnxResult> ret = postprocess();
  __TOC__(postprocess)
  __TOC__(total)
  return ret;
}
