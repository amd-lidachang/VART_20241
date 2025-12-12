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
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
cv::Mat process_result(cv::Mat& m1, const vitis::ai::BTSResult& result,
                       bool is_jpeg) {
  cv::Mat image;
  cv::resize(m1, image, cv::Size{result.width, result.height});
  FILE *fp = fopen("bts_pt_batch_0.bin", "wb");
  if(fp == nullptr) {
          std::cout<<"fail to open!";
  }
  if (is_jpeg) {
    for (auto i = 0; i < result.height; ++i) {
      for (auto j = 0; j < result.width; ++j) {
       // std::cout << result.depth_mat.at<uint16_t>(i, j) << " ";
        uint16_t data = result.depth_mat.at<uint16_t>(i, j);
        fwrite(&data, sizeof(uint16_t), 1, fp);
      }
      // std::cout << std::endl;
    }
  }
  return image;
}
