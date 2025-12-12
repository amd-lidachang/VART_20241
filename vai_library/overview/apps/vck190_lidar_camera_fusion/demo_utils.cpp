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

#include "./demo_utils.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/profiling.hpp>

using std::cos;
using std::sin;
namespace vitis {
namespace ai {
namespace pointpainting {

DEF_ENV_PARAM(DEBUG_DRAW, "0");
DEF_ENV_PARAM(DEBUG_CAM_DRAW, "0");

void print_vector2d(const V2F& v, const std::string& name) {
  std::cout << name << " size: " << v.size() << std::endl;
  for (auto j = 0u; j < v.size(); ++j) {
    for (auto k = 0u; k < v[j].size(); ++k) {
      std::cout << v[j][k] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void print_mat2d(const cv::Mat& mat, const std::string& name) {
  std::cout << name << " size: " << mat.rows << " * " << mat.cols << std::endl;
  for (auto j = 0; j < mat.rows; ++j) {
    for (auto k = 0; k < mat.cols; ++k) {
      std::cout << mat.at<float>(j, k) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

std::vector<std::vector<int>> make_bv_feature(
    const std::vector<int>& point_range, const std::vector<float> bbox_3d,
    float resolution, int width, int height) {
  std::vector<std::vector<int>> result(4);
  assert(point_range.size() >= 6);
  if (ENV_PARAM(DEBUG_DRAW)) {
    std::cout << "box:";
    for (auto i = 0u; i < bbox_3d.size(); ++i) {
      std::cout << bbox_3d[i] << " ";
    }
    std::cout << std::endl;
  }
  auto corners = generate8Corners(bbox_3d);
  if (ENV_PARAM(DEBUG_DRAW)) {
    print_vector2d(corners, "corners");
  }
  cv::Mat corners_img = cv::Mat::zeros(4, 2, CV_32FC1);
  for (auto i = 0; i < corners_img.rows; ++i) {
    corners_img.at<float>(i, 0) = (corners[i][0] - point_range[0]) / resolution;
    corners_img.at<float>(i, 1) = (corners[i][1] - point_range[1]) / resolution;
    corners_img.at<float>(i, 0) = width - corners_img.at<float>(i, 0);
    corners_img.at<float>(i, 1) = height - corners_img.at<float>(i, 1);
  }
  if (ENV_PARAM(DEBUG_DRAW)) {
    print_mat2d(corners_img, "corners_img");
  }
  for (auto i = 0; i < 4; ++i) {
    result[i] = std::vector<int>{(int)corners_img.at<float>(i, 0),
                                 (int)corners_img.at<float>(i, 1),
                                 (int)corners_img.at<float>((i + 1) % 4, 0),
                                 (int)corners_img.at<float>((i + 1) % 4, 1)};
  }
  return result;
}

V2F roty(float x) {
  return V2F{{cos(x), 0, sin(x)}, {0, 1, 0}, {-sin(x), 0, cos(x)}};
}

V2F rotz(float t) {
  auto c = cos(t);
  auto s = sin(t);
  return V2F{{c, -s, 0}, {s, c, 0}, {0, 0, 1}};
}

// ibox:  xyz lhw a
V2F generate8CornersKitti(const V1F& ibox) {
  V2F offset = {{0.5, 0, 0.5},    {-0.5, 0, 0.5}, {-0.5, -1, 0.5},
                {0.5, -1, 0.5},   {0.5, 0, -0.5}, {-0.5, 0, -0.5},
                {-0.5, -1, -0.5}, {0.5, -1, -0.5}};
  for (unsigned int i = 0; i < offset.size(); i++) {
    offset[i][0] *= ibox[3];
    offset[i][1] *= ibox[4];
    offset[i][2] *= ibox[5];
  }

  // dot () .T
  V2F roty_v = roty(ibox[6]);
  V2F corners(8, V1F(3, 0));

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        corners[i][j] += roty_v[j][k] * offset[i][k];
      }
      corners[i][j] += ibox[j];
    }
  }

  return corners;
}

// x, y, z, dx, dy, dz, rz
V2F generate8Corners(const V1F& ibox) {
  V2F offset = {{0.5, 0.5, 0}, {-0.5, 0.5, 0}, {-0.5, -0.5, 0}, {0.5, -0.5, 0},
                {0.5, 0.5, 1}, {-0.5, 0.5, 1}, {-0.5, -0.5, 1}, {0.5, -0.5, 1}};
  for (unsigned int i = 0; i < offset.size(); i++) {
    offset[i][0] *= ibox[3];
    offset[i][1] *= ibox[4];
    offset[i][2] *= ibox[5];
  }
  if (ENV_PARAM(DEBUG_DRAW)) {
    print_vector2d(offset, "offset");
  }

  V2F rotz_v = rotz(ibox[6]);
  cv::Mat rotz_mat = vector2mat(rotz_v);
  cv::Mat offset_mat = vector2mat(offset);
  offset_mat = rotz_mat * offset_mat.t();
  offset_mat = offset_mat.t();

  V2F corners(8, V1F(3, 0));
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 3; j++) {
      corners[i][j] = offset_mat.at<float>(i, j) + ibox[j];
    }
  }
  return corners;
}

cv::Mat project_3d_pts(const cv::Mat& corners, const cv::Mat& intrinsic) {
  if (ENV_PARAM(DEBUG_CAM_DRAW)) {
    print_mat2d(corners, "corners mat");
    print_mat2d(intrinsic, "intrinsic mat");
  }
  cv::Mat pc_img = cv::Mat::zeros(8, 3, CV_32FC1);
  cv::Mat proj_pts = (intrinsic * corners.t()).t();
  for (auto i = 0; i < pc_img.rows; ++i) {
    pc_img.at<float>(i, 0) =
        proj_pts.at<float>(i, 0) / proj_pts.at<float>(i, 2);
    pc_img.at<float>(i, 1) =
        proj_pts.at<float>(i, 1) / proj_pts.at<float>(i, 2);
    pc_img.at<float>(i, 2) = proj_pts.at<float>(i, 2);
  }
  return pc_img;
}

// V2F project_3d_pts(V2F& corners, const V2F& p2rect) {
//  if (ENV_PARAM(DEBUG_CAM_DRAW)) {
//    print_vector2d(corners, "corners");
//    print_vector2d(p2rect, "intrinsic");
//  }
//  V2F proj_pts(8, V1F(4, 0));
//
//  for (int i = 0; i < 8; i++) {
//    for (int j = 0; j < 4; j++) {
//      for (int k = 0; k < 4; k++) {
//        proj_pts[i][j] += p2rect[j][k] * (k == 3 ? 1.0 : corners[i][k]);
//      }
//    }
//    proj_pts[i][0] /= proj_pts[i][2];
//    proj_pts[i][1] /= proj_pts[i][2];
//    proj_pts[i][2] = proj_pts[i][2];
//  }
//  return proj_pts;
//}

void draw_projected_box3d(cv::Mat& img, const V2F& qs, int label) {
  int thickness = 1;
  float scale = 1.0;

  if (img.rows != 900) {
    scale = ((float)img.rows / 900);
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_CAM_DRAW)) << "cam scale:" << scale;
  for (int k = 0; k < 4; k++) {
    std::vector<int> vi = {k, k + 4, k};
    std::vector<int> vj = {(k + 1) % 4, (k + 1) % 4 + 4, k + 4};
    for (int kk = 0; kk < 3; kk++) {
      if (img.rows && img.cols) {
        cv::line(
            img,
            cv::Point2f(int(qs[vi[kk]][0] * scale), int(qs[vi[kk]][1] * scale)),
            cv::Point2f(int(qs[vj[kk]][0] * scale), int(qs[vj[kk]][1] * scale)),
            // cv::Scalar(0,255,255),
            vscalar[label], thickness);

        LOG_IF(INFO, ENV_PARAM(DEBUG_CAM_DRAW))
            << "line from:(" << qs[vi[kk]][0] * scale << ","
            << qs[vj[kk]][1] * scale << ")"
            << " to:(" << qs[vj[kk]][0] * scale << "," << qs[vj[kk]][1] * scale
            << ")";
      } else {
        LOG_IF(INFO, ENV_PARAM(DEBUG_CAM_DRAW))
            << "line from:(" << qs[vi[kk]][0] << "*" << scale << ","
            << qs[vj[kk]][1] << "*" << scale << ")"
            << " to:(" << qs[vj[kk]][0] << "*" << scale << "," << qs[vj[kk]][1]
            << "*" << scale << ")";
      }
    }
  }
}

cv::Mat make_l2c_mat(const std::array<float, 9>& s2l_r,
                     const std::array<float, 3>& s2l_t) {
  cv::Mat c2l_mat = cv::Mat::eye(4, 4, CV_32FC1);
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      c2l_mat.at<float>(i, j) = s2l_r[i * 3 + j];
    }
    c2l_mat.at<float>(i, 3) = s2l_t[i];
  }
  cv::Mat l2c_mat = c2l_mat.inv();
  return l2c_mat;
}

cv::Mat make_intrinsic_mat(const std::array<float, 9>& cam_intr) {
  cv::Mat intr_mat = cv::Mat::zeros(3, 4, CV_32FC1);
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      intr_mat.at<float>(i, j) = cam_intr[i * 3 + j];
    }
  }
  return intr_mat;
}
cv::Mat vector2mat(const V2F& v) {
  assert(v.size() > 0);
  auto H = v.size();
  auto W = v[0].size();
  cv::Mat result = cv::Mat::zeros(H, W, CV_32FC1);
  for (auto h = 0u; h < H; ++h) {
    for (auto w = 0u; w < W; ++w) {
      result.at<float>(h, w) = v[h][w];
    }
  }
  return result;
}

V2F mat2vector(const cv::Mat& mat) {
  auto H = mat.rows;
  auto W = mat.cols;
  V2F result(H, std::vector<float>(W));
  for (auto h = 0; h < H; ++h) {
    for (auto w = 0; w < W; ++w) {
      result[h][w] = mat.at<float>(h, w);
    }
  }
  return result;
}

bool check_back(const V2F& corners_on_cam, unsigned axis) {
  if (corners_on_cam.size() == 0) {
    return false;
  }
  for (auto i = 0u; i < corners_on_cam.size(); ++i) {
    if (corners_on_cam[i][axis] < 0) {
      return true;
    }
  }
  return false;
}

V2F make_corners_on_cam(const V2F& corners, const V2F& l2c) {
  V2F homo_corners;
  homo_corners.resize(corners.size());
  for (auto i = 0u; i < homo_corners.size(); ++i) {
    homo_corners[i].resize(4);
    std::copy(corners[i].begin(), corners[i].begin() + 3,
              homo_corners[i].begin());
    homo_corners[i][3] = 1.0;
  }
  auto homo_mat = vector2mat(homo_corners);
  auto l2c_mat = vector2mat(l2c);
  cv::Mat corners_on_cam_mat = l2c_mat * homo_mat.t();

  corners_on_cam_mat = corners_on_cam_mat.t();
  auto corners_on_cam = mat2vector(corners_on_cam_mat);

  return corners_on_cam;
}

void draw_proj_box(const V1F& ibox, cv::Mat& img, const V2F& l2c,
                   const V2F& intrinsic, int label) {
  // V2F corners = generate8CornersKitti(ibox);
  if (ENV_PARAM(DEBUG_DRAW)) {
    std::cout << "box:";
    for (auto i = 0u; i < ibox.size(); ++i) {
      std::cout << ibox[i] << " ";
    }
    std::cout << std::endl;
  }

  V2F corners = generate8Corners(ibox);
  if (corners.empty()) {
    return;
  }

  if (ENV_PARAM(DEBUG_CAM_DRAW)) {
    print_vector2d(l2c, "l2c");
    print_vector2d(intrinsic, "intrinsic");
    print_vector2d(corners, "corners");
  }

  auto corners_on_cam = make_corners_on_cam(corners, l2c);
  if (ENV_PARAM(DEBUG_CAM_DRAW)) {
    print_vector2d(corners_on_cam, "corners_on_cam");
  }

  if (check_back(corners_on_cam, 2)) {
    return;
  }
  auto corners_on_cam_mat = vector2mat(corners_on_cam);
  auto intrinsic_mat = vector2mat(intrinsic);
  cv::Mat proj_pts_mat = project_3d_pts(corners_on_cam_mat, intrinsic_mat);
  if (ENV_PARAM(DEBUG_CAM_DRAW)) {
    // print_vector2d(proj_pts, "proj_pts");
    print_mat2d(proj_pts_mat, "proj_pts_mat");
  }
  auto proj_pts = mat2vector(proj_pts_mat);
  draw_projected_box3d(img, proj_pts, label);
}





}  // namespace pointpainting
}  // namespace ai
}  // namespace vitis
