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

#include "vitis/ai/preprocess_yolo.hpp"

#include <opencv2/imgproc.hpp>
#include <vector>

#include "utils_yolov2.hpp"

namespace vitis {
namespace ai {
namespace math {

void convertInputImage(const cv::Mat& frame, const int width, const int height,
                       const int channel, const float scale, int8_t* data) {
  int size = width * height * channel;
  image img_new = load_image_cv(frame);
  image img_yolo = letterbox_image(img_new, width, height);

  std::vector<float> bb(size);
  for (int b = 0; b < height; ++b) {
    for (int c = 0; c < width; ++c) {
      for (int a = 0; a < channel; ++a) {
        bb[b * width * channel + c * channel + a] =
            img_yolo.data[a * height * width + b * width + c];
      }
    }
  }
  for (int i = 0; i < size; ++i) {
    data[i] = int(bb.data()[i] * scale);
    if (data[i] < 0) data[i] = 127;
  }
  free_image(img_new);
  free_image(img_yolo);
}

void convertInputImage(const cv::Mat& frame, const int width, const int height,
                       const int channel, const float scale, float* data) {
  int size = width * height * channel;
  image img_new = load_image_cv(frame);
  image img_yolo = letterbox_image(img_new, width, height);

  std::vector<float> bb(size);

  for (int a = 0; a < channel; ++a) {
    for (int b = 0; b < height; ++b) {
      for (int c = 0; c < width; ++c) {
        bb[a * width * height + b * width + c] =
            img_yolo.data[a * height * width + b * width + c];
      }
    }
  }

  for (int i = 0; i < size; ++i) {
    data[i] = bb.data()[i];
    if (data[i] < 0) data[i] = (float)scale;
  }

  free_image(img_new);
  free_image(img_yolo);
}

cv::Mat letterbox_v2(const cv::Mat& im, const int w, const int h) {
  float scale = std::min((float)w / (float)im.cols, (float)h / (float)im.rows);
  int new_w = im.cols * scale;
  int new_h = im.rows * scale;
  cv::Mat img_res;
  cv::resize(im, img_res, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  cv::Mat new_img(cv::Size(w, h), CV_8UC3, cv::Scalar(128, 128, 128));
  int x = (w - new_w) / 2;
  int y = (h - new_h) / 2;
  auto rect = cv::Rect{x, y, new_w, new_h};
  img_res.copyTo(new_img(rect));
  return new_img;
}

cv::Mat letterbox_v4(const cv::Mat& im, const int w, const int h) {
  float scale = std::min((float)w / (float)im.cols, (float)h / (float)im.rows);
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

cv::Mat letterbox_tf(const cv::Mat& im, const int w, const int h) {
  float scale = std::min((float)w / (float)im.cols, (float)h / (float)im.rows);
  int new_w = im.cols * scale;
  int new_h = im.rows * scale;
  cv::Mat img_res;
  cv::resize(im, img_res, cv::Size(new_w, new_h));

  cv::Mat new_img(cv::Size(w, h), CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat t_mat = cv::Mat::zeros(2, 3, CV_32FC1);
  t_mat.at<float>(0, 0) = 1;
  t_mat.at<float>(0, 2) = (w - new_w) / 2;
  t_mat.at<float>(1, 1) = 1;
  t_mat.at<float>(1, 2) = (h - new_h) / 2;
  cv::warpAffine(img_res, new_img, t_mat, new_img.size());
  return new_img;
}

void letterbox_v6(const cv::Mat& im, const int w, const int h,
                  const int load_size, cv::Mat& om, float& scale, int& left,
                  int& top) {
  cv::Mat img_res;
  float r = (float)load_size / (float)std::max(im.cols, im.rows);
  if (r != 1) {
    cv::resize(im, img_res, cv::Size(im.cols * r, im.rows * r), 0, 0,
               r < 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
  }
  float dw = (w - img_res.cols) / 2.0f;
  float dh = (h - img_res.rows) / 2.0f;
  top = int(round(dh - 0.1));
  int bottom = int(round(dh + 0.1));
  left = int(round(dw - 0.1));
  int right = int(round(dw + 0.1));
  cv::copyMakeBorder(img_res, om, top, bottom, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  scale = r;
}

void letterbox_v8(const cv::Mat input_image, cv::Mat& output_image,
                  const int height, const int width, float& scale, int& left,
                  int& top) {
  cv::Mat image_tmp;

  scale = std::min(float(width) / input_image.cols,
                   float(height) / input_image.rows);
  scale = std::min(scale, 1.0f);
  int unpad_w = round(input_image.cols * scale);
  int unpad_h = round(input_image.rows * scale);
  image_tmp = input_image.clone();

  if (input_image.size() != cv::Size(unpad_w, unpad_h)) {
    cv::resize(input_image, image_tmp, cv::Size(unpad_w, unpad_h),
               cv::INTER_LINEAR);
  }

  float dw = (width - unpad_w) / 2.0f;
  float dh = (height - unpad_h) / 2.0f;

  top = round(dh - 0.1);
  int bottom = round(dh + 0.1);
  left = round(dw - 0.1);
  int right = round(dw + 0.1);

  cv::copyMakeBorder(image_tmp, output_image, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  return;
}

void letterbox_vx(const cv::Mat& im, const int w, const int h, cv::Mat& om,
                  float& scale) {
  scale = std::min((float)w / (float)im.cols, (float)h / (float)im.rows);
  cv::Mat img_res;
  if (im.size() != cv::Size(w, h)) {
    cv::resize(im, img_res, cv::Size(im.cols * scale, im.rows * scale), 0, 0,
               cv::INTER_LINEAR);
    auto dw = w - img_res.cols;
    auto dh = h - img_res.rows;
    if (dw > 0 || dh > 0) {
      om = cv::Mat(cv::Size(w, h), CV_8UC3, cv::Scalar(128, 128, 128));
      copyMakeBorder(img_res, om, 0, dh, 0, dw, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
    } else {
      om = img_res;
    }
  } else {
    om = im;
    scale = 1.0;
  }
}

cv::Mat letterbox_ofa(const cv::Mat& im, const int w, const int h) {
  float scale = std::min((float)w / (float)im.cols, (float)h / (float)im.rows);

  int new_w = round(im.cols * scale);
  int new_h = round(im.rows * scale);

  cv::Mat img_res;
  if (im.size() != cv::Size(new_w, new_h)) {
    cv::resize(im, img_res, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  } else {
    img_res = im;
  }
  auto dw = float(w - new_w) / 2.0f;
  auto dh = float(h - new_h) / 2.0f;

  cv::Mat new_img(cv::Size(w, h), CV_8UC3, cv::Scalar(128, 128, 128));
  copyMakeBorder(img_res, new_img, int(round(dh - 0.1)), int(round(dh + 0.1)),
                 int(round(dw - 0.1)), int(round(dw + 0.1)),
                 cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

  return new_img;
}

}  // namespace math
}  // namespace ai
}  // namespace vitis

