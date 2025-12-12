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
#include <glog/logging.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo3.hpp>
#include <vitis/ai/facedetect.hpp>
#include <vitis/ai/lanedetect.hpp>
#include <vitis/ai/multitask.hpp>
#include <vitis/ai/posedetect.hpp>
#include <vitis/ai/ssd.hpp>
using namespace std;

namespace vitis {
namespace ai {
struct SSDPoseDetect {
  static std::unique_ptr<SSDPoseDetect> create();
  SSDPoseDetect();
  std::vector<std::pair<std::vector<vitis::ai::PoseDetectResult>,
                        vitis::ai::FaceDetectResult>>
  run(const std::vector<cv::Mat>& input_image);
  int getInputWidth();
  int getInputHeight();
  size_t get_input_batch() const {
    return 1u;  // std::min(ssd_->get_input_batch(),
                // face_detect_->get_input_batch());
  };

 private:
  std::unique_ptr<vitis::ai::SSD> ssd_;
  std::unique_ptr<vitis::ai::PoseDetect> pose_detect_;
  std::unique_ptr<vitis::ai::FaceDetect> face_detect_;
};
// A factory function to get a instance of derived classes of class
std::unique_ptr<SSDPoseDetect> SSDPoseDetect::create() {
  return std::unique_ptr<SSDPoseDetect>(new SSDPoseDetect());
}
int SSDPoseDetect::getInputWidth() { return ssd_->getInputWidth(); }
int SSDPoseDetect::getInputHeight() { return ssd_->getInputHeight(); }

SSDPoseDetect::SSDPoseDetect()
    : ssd_{vitis::ai::SSD::create("ssd_pedestrian_pruned_0_97", true)},
      face_detect_{vitis::ai::FaceDetect::create("densebox_640_360", true)},
      pose_detect_{vitis::ai::PoseDetect::create("sp_net")} {}

std::vector<std::pair<std::vector<vitis::ai::PoseDetectResult>,
                      vitis::ai::FaceDetectResult>>
SSDPoseDetect::run(const std::vector<cv::Mat>& input_images) {
  std::vector<std::pair<std::vector<vitis::ai::PoseDetectResult>,
                        vitis::ai::FaceDetectResult>>
      results;
  // run ssd
  auto ssd_results_batch = ssd_->run(input_images);
  auto face_results_batch = face_detect_->run(input_images);
  CHECK_EQ(ssd_results_batch.size(), face_results_batch.size());
  auto k = 0u;
  for (auto& ssd_results : ssd_results_batch) {
    std::vector<vitis::ai::PoseDetectResult> mt_results;
    for (auto& box : ssd_results.bboxes) {
      if (1) {
        DLOG(INFO) << "box.x " << box.x << " "            //
                   << "box.y " << box.y << " "            //
                   << "box.width " << box.width << " "    //
                   << "box.height " << box.height << " "  //
                   << "box.score " << box.score << " "    //
            ;
      }
      // int label = box.label;
      int xmin = box.x * input_images[k].cols;
      int ymin = box.y * input_images[k].rows;
      int xmax = xmin + box.width * input_images[k].cols;
      int ymax = ymin + box.height * input_images[k].rows;
      float confidence = box.score;
      if (confidence < 0.55) continue;
      xmin = std::min(std::max(xmin, 0), input_images[k].cols);
      xmax = std::min(std::max(xmax, 0), input_images[k].cols);
      ymin = std::min(std::max(ymin, 0), input_images[k].rows);
      ymax = std::min(std::max(ymax, 0), input_images[k].rows);
      cv::Rect roi = cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
      cv::Mat sub_img = input_images[k](roi);
      // process each result of ssd detection
      auto single_result = pose_detect_->run(sub_img);
      for (size_t i = 0; i < 28; i = i + 2) {
        ((float*)&single_result.pose14pt)[i] =
            ((float*)&single_result.pose14pt)[i] * sub_img.cols;
        ((float*)&single_result.pose14pt)[i] =
            (((float*)&single_result.pose14pt)[i] + xmin) /
            input_images[k].cols;
        ((float*)&single_result.pose14pt)[i + 1] =
            ((float*)&single_result.pose14pt)[i + 1] * sub_img.rows;
        ((float*)&single_result.pose14pt)[i + 1] =
            (((float*)&single_result.pose14pt)[i + 1] + ymin) /
            input_images[k].rows;
      }
      mt_results.emplace_back(single_result);
    }
    results.push_back(make_pair(mt_results, face_results_batch[k]));
    k = k + 1;
  }
  return results;
}
}  // namespace ai
}  // namespace vitis

#ifndef HAVE_EIGEN
#define HAVE_EIGEN 0
#endif
#if HAVE_EIGEN
#include <eigen3/Eigen/Dense>
// Overlay the original image with the result
// Eigen Optimized version
static void overLay1(cv::Mat& src1, const cv::Mat& src2) {
  const int imsize = src1.cols * src2.rows * 3;
  Eigen::Map<Eigen::Matrix<uchar, -1, 1>> data1(const_cast<uchar*>(src1.data),
                                                imsize);
  Eigen::Map<Eigen::Matrix<uchar, -1, 1>> data2(const_cast<uchar*>(src2.data),
                                                imsize);
  data1 = data1 / 2 + data2 / 2;
}
#else
// c version
static void overLay1(cv::Mat& src1, const cv::Mat& src2) {
  const int imsize = src1.cols * src2.rows * 3;
  for (int i = 0; i < imsize; ++i) {
    src1.data[i] = src1.data[i] / 2 + src2.data[i] / 2;
  }
}
#endif
// This function is used to process the multitask result and show on the image
static cv::Mat process_result_multitask(cv::Mat& m1,
                                        vitis::ai::MultiTaskResult& result,
                                        bool is_jpeg) {
  (void)process_result_multitask;
  cv::Mat image;
  // Overlay segmentation result to the original image
  if (false) {
    cv::resize(m1, image, result.segmentation.size());
    overLay1(image, result.segmentation);
  } else {
    cv::resize(result.segmentation, image, m1.size());
    overLay1(image, m1);
  }
  // Draw detection results
  for (auto& r : result.vehicle) {
    LOG_IF(INFO, is_jpeg) << r.label << " " << r.x << " " << r.y << " "
                          << r.width << " " << r.height << " " << r.angle;
    int xmin = r.x * image.cols;
    int ymin = r.y * image.rows;

    int width = r.width * image.cols;
    int height = r.height * image.rows;
    cv::rectangle(image, cv::Rect_<int>(xmin, ymin, width, height),
                  cv::Scalar(185, 181, 178), 2, 1, 0);
  }
  return image;
}

using namespace cv;

// This function is used to process the roadline result and show on the image
cv::Mat process_result_roadline(cv::Mat& image,
                                const vitis::ai::RoadLineResult& result,
                                bool is_jpeg) {
  std::vector<int> color1 = {0, 255, 0, 0, 100, 255};
  std::vector<int> color2 = {0, 0, 255, 0, 100, 255};
  std::vector<int> color3 = {0, 0, 0, 255, 100, 255};

  LOG_IF(INFO, is_jpeg) << "lines.size " << result.lines.size() << " ";
  for (auto& line : result.lines) {
    LOG_IF(INFO, is_jpeg) << "line.points_cluster.size() "
                          << line.points_cluster.size() << " ";
    std::vector<cv::Point> points_poly = line.points_cluster;
    int type = line.type < 5 ? line.type : 5;
    if (type == 2 && points_poly[0].x < image.rows * 0.5) continue;
    cv::polylines(image, points_poly, false,
                  cv::Scalar(color1[type], color2[type], color3[type]), 3,
                  CV_AA, 0);
  }
  return image;
}

static inline void DrawLine(Mat& img, Point2f point1, Point2f point2,
                            Scalar colour, int thickness, float scale_w,
                            float scale_h) {
  if ((point1.x * img.cols > scale_w || point1.y * img.rows > scale_h) &&
      (point2.x * img.cols > scale_w || point2.y * img.rows > scale_h))
    cv::line(img, Point2f(point1.x * img.cols, point1.y * img.rows),
             Point2f(point2.x * img.cols, point2.y * img.rows), colour,
             thickness);
}

static void DrawLines(Mat& img,
                      const vitis::ai::PoseDetectResult::Pose14Pt& results) {
  float scale_w = 1;
  float scale_h = 1;

  float mark = 5.f;
  float mark_w = mark * scale_w;
  float mark_h = mark * scale_h;
  std::vector<Point2f> pois(14);
  for (size_t i = 0; i < pois.size(); ++i) {
    pois[i].x = ((float*)&results)[i * 2] * img.cols;
    // std::cout << ((float*)&results)[i * 2] << " " << ((float*)&results)[i * 2
    // + 1] << std::endl;
    pois[i].y = ((float*)&results)[i * 2 + 1] * img.rows;
  }
  for (size_t i = 0; i < pois.size(); ++i) {
    circle(img, pois[i], 3, Scalar::all(255));
  }
  DrawLine(img, results.right_shoulder, results.right_elbow, Scalar(255, 0, 0),
           2, mark_w, mark_h);
  DrawLine(img, results.right_elbow, results.right_wrist, Scalar(255, 0, 0), 2,
           mark_w, mark_h);
  DrawLine(img, results.right_hip, results.right_knee, Scalar(255, 0, 0), 2,
           mark_w, mark_h);
  DrawLine(img, results.right_knee, results.right_ankle, Scalar(255, 0, 0), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_shoulder, results.left_elbow, Scalar(0, 0, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_elbow, results.left_wrist, Scalar(0, 0, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_hip, results.left_knee, Scalar(0, 0, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_knee, results.left_ankle, Scalar(0, 0, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.head, results.neck, Scalar(0, 255, 255), 2, mark_w,
           mark_h);
  DrawLine(img, results.right_shoulder, results.neck, Scalar(0, 255, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_shoulder, results.neck, Scalar(0, 255, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.right_shoulder, results.right_hip, Scalar(0, 255, 255),
           2, mark_w, mark_h);
  DrawLine(img, results.left_shoulder, results.left_hip, Scalar(0, 255, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.right_hip, results.left_hip, Scalar(0, 255, 255), 2,
           mark_w, mark_h);
}

cv::Mat process_result_pose_detect(cv::Mat& image,
                                   const vitis::ai::PoseDetectResult& results,
                                   bool is_jpeg) {
  std::vector<float> pose14pt_arry((float*)&results.pose14pt,
                                   (float*)&results.pose14pt + 28);
  for (size_t i = 0; i < pose14pt_arry.size(); i = i + 2) {
    LOG_IF(INFO, is_jpeg) << "(" << pose14pt_arry[i] << ","
                          << pose14pt_arry[i + 1] << ")";
  }
  DrawLines(image, results.pose14pt);
  return image;
}

static cv::Mat process_result_pose_detect_with_ssd(
    cv::Mat& image,
    const pair<std::vector<vitis::ai::PoseDetectResult>,
               vitis::ai::FaceDetectResult>& result,
    bool is_jpeg) {
  (void)process_result_pose_detect_with_ssd;

  for (auto& result : result.first) {
    process_result_pose_detect(image, result, is_jpeg);
  }
  for (const auto& r : result.second.rects) {
    cv::rectangle(image,
                  cv::Rect{cv::Point(r.x * image.cols, r.y * image.rows),
                           cv::Size{(int)(r.width * image.cols),
                                    (int)(r.height * image.rows)}},
                  0xff);
  }

  // image.copyTo(canvas(Rect(0, 0, image.cols, image.rows)));
  return image;
}
using namespace vitis::ai;
int main(int argc, char* argv[]) {
  // set the layout
  //
  int seg_px = 0;  // 100 his
  int seg_py = 0;  // 252 his
  // assign to Lvalue : static std::vector<cv::Rect> rects, the coordinates of
  // each window
  gui_layout() = {
      {seg_px, seg_py, 512, 288},
      {seg_px + 512, seg_py, 512, 288},
      {seg_px, seg_py + 288, 512, 288},
      {seg_px + 512, seg_py + 288, 512, 288},
      {1024, 0, 640, 480},
      {1024, 480, 640, 360},
  };
  // assign to Lvalue : set background image
  //
  gui_background() = cv::imread("/usr/share/weston/logo.jpg");
  // init each dpu filter and process instance, using video demo framework
  return vitis::ai::main_for_video_demo_multiple_channel(
      argc, argv,
      {
          DpuFilterDesciption{string("multitask"),
                              [] {
                                return vitis::ai::create_dpu_filter(
                                    [] {
                                      return vitis::ai::MultiTask8UC3::create(
                                          "multi_task");
                                    },
                                    process_result_multitask);
                              },
                              4},
          DpuFilterDesciption{std::string("roadline"),
                              [] {
                                return vitis::ai::create_dpu_filter(
                                    [] {
                                      return vitis::ai::RoadLine::create(
                                          "vpgnet_pruned_0_99");
                                    },
                                    process_result_roadline);
                              },
                              1},
          DpuFilterDesciption{std::string("pose"),
                              [] {
                                return vitis::ai::create_dpu_filter(
                                    [] {
                                      return vitis::ai::SSDPoseDetect::create();
                                    },
                                    process_result_pose_detect_with_ssd);
                              },
                              1},
      });
}
