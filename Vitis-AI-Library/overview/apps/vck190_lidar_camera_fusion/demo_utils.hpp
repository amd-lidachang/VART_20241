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

#include <opencv2/imgcodecs.hpp>
#include <vector>

namespace vitis {
namespace ai {
namespace pointpainting {

using V1F = std::vector<float>;
using V2F = std::vector<V1F>;

static const std::vector<cv::Scalar> vscalar = {
    cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
    cv::Scalar(0, 0, 255),   cv::Scalar(255, 0, 0),   cv::Scalar(0, 255, 0)};

std::vector<std::vector<int>> make_bv_feature(
    const std::vector<int>& point_range, const std::vector<float> bbox_3d,
    float resolution, int width, int height);

V2F generate8CornersKitti(const V1F& ibox);
V2F generate8Corners(const V1F& ibox);
void draw_proj_box(const V1F& ibox, cv::Mat& img, const V2F& p2rect, int label);
void draw_proj_box(const V1F& ibox, cv::Mat& img, const V2F& l2c,
                   const V2F& intrinsic, int label);

V2F mat2vector(const cv::Mat& mat);
cv::Mat vector2mat(const V2F& v);

cv::Mat make_l2c_mat(const std::array<float, 9>& s2l_r,
                     const std::array<float, 3>& s2l_t);
cv::Mat make_intrinsic_mat(const std::array<float, 9>& cam_intr);


}  // namespace pointpainting
}  // namespace ai
}  // namespace vitis
