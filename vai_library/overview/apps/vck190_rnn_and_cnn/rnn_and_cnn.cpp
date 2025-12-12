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

#include <Python.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <linux/fb.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/facedetect.hpp>
#include <vitis/ai/lanedetect.hpp>
#include <vitis/ai/multitask.hpp>
#include <vitis/ai/posedetect.hpp>
#include <vitis/ai/ssd.hpp>

#include "demo.hpp"
using namespace std;
using namespace cv;

DEF_ENV_PARAM(ENABLE_RNN, "1")

namespace vitis {
namespace ai {
struct SSDPoseDetect {
  static std::unique_ptr<SSDPoseDetect> create();
  SSDPoseDetect();
  std::vector<std::vector<vitis::ai::PoseDetectResult>> run(
      const std::vector<cv::Mat>& input_image);
  int getInputWidth();
  int getInputHeight();

 private:
  std::unique_ptr<vitis::ai::SSD> ssd_;
  std::unique_ptr<vitis::ai::PoseDetect> pose_detect_;
};
// A factory function to get a instance of derived classes of class
std::unique_ptr<SSDPoseDetect> SSDPoseDetect::create() {
  return std::unique_ptr<SSDPoseDetect>(new SSDPoseDetect());
}
int SSDPoseDetect::getInputWidth() { return ssd_->getInputWidth(); }
int SSDPoseDetect::getInputHeight() { return ssd_->getInputHeight(); }

SSDPoseDetect::SSDPoseDetect()
    : ssd_{vitis::ai::SSD::create("ssd_pedestrian_pruned_0_97", true)},
      pose_detect_{vitis::ai::PoseDetect::create("sp_net")} {}

std::vector<std::vector<vitis::ai::PoseDetectResult>> SSDPoseDetect::run(
    const std::vector<cv::Mat>& input_images) {
  std::vector<std::vector<vitis::ai::PoseDetectResult>> results;
  for (auto k = 0u; k < input_images.size(); k++) {
    std::vector<vitis::ai::PoseDetectResult> mt_results;
    cv::Mat image;
    auto size = cv::Size(ssd_->getInputWidth(), ssd_->getInputHeight());
    if (size != input_images[k].size()) {
      cv::resize(input_images[k], image, size);
    } else {
      image = input_images[k];
    }
    // run ssd
    auto ssd_results = ssd_->run(image);

    for (auto& box : ssd_results.bboxes) {
      if (0)
        DLOG(INFO) << "box.x " << box.x << " "            //
                   << "box.y " << box.y << " "            //
                   << "box.width " << box.width << " "    //
                   << "box.height " << box.height << " "  //
                   << "box.score " << box.score << " "    //
            ;
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
    results.push_back(mt_results);
  }
  return results;
}
}  // namespace ai
}  // namespace vitis

static void overLay1(cv::Mat& src1, const cv::Mat& src2) {
  const int imsize = src1.cols * src2.rows * 3;
  for (int i = 0; i < imsize; ++i) {
    src1.data[i] = src1.data[i] / 2 + src2.data[i] / 2;
  }
}
static std::vector<cv::Mat> process_result_multitask(
    std::vector<cv::Mat>& m1s,
    const std::vector<vitis::ai::MultiTaskResult>& results, bool is_jpeg) {
  (void)process_result_multitask;
  std::vector<cv::Mat> images(m1s.size());
  for (auto i = 0u; i < m1s.size(); i++) {
    cv::Mat m1 = m1s[i];
    auto result = results[i];
    cv::Mat image;
    // Overlay segmentation result to the original image
    cv::resize(result.segmentation, image, m1.size());
    overLay1(image, m1);
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
    images[i] = image;
  }

  return images;
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

static std::vector<cv::Mat> process_result_pose_detect_with_ssd(
    std::vector<cv::Mat>& images,
    const std::vector<std::vector<vitis::ai::PoseDetectResult>>& all_results,
    bool is_jpeg) {
  (void)process_result_pose_detect_with_ssd;

  for (auto i = 0u; i < images.size(); i++) {
    for (auto& result : all_results[i]) {
      DrawLines(images[i], result.pose14pt);
    }
  }

  return images;
}

static std::vector<cv::Mat> process_result_facedetect(
    std::vector<cv::Mat>& images,
    std::vector<vitis::ai::FaceDetectResult> results, bool is_jpeg) {
  for (auto im = 0u; im < images.size(); im++) {
    for (const auto& r : results[im].rects) {
      cv::rectangle(
          images[im],
          cv::Rect{cv::Point(r.x * images[im].cols, r.y * images[im].rows),
                   cv::Size{(int)(r.width * images[im].cols),
                            (int)(r.height * images[im].rows)}},
          cv::Scalar(255, 0, 0), 2, 2, 0);
    }
  }
  return images;
}

// Execute third party filter
struct CNNThread : public vitis::ai::MyThread {
  CNNThread(std::string videofile, std::string wavfile)
      : MyThread{}, name_{std::string("CNNThread")}, all_text_("") {
    gui_thread4_ = std::make_unique<vitis::ai::GuiThread>(4);
    queue_out4_ = gui_thread4_->getQueue();
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT " << name_;
    auto& layout4 = gui_layout()[4];
    background_ = cv::Mat(layout4.dst_height, layout4.dst_width, CV_8UC3,
                          cv::Scalar(255, 255, 255, 255));
    background_.setTo(255);

    vitis::ai::FrameInfo frame;
    frame.mat = background_.clone();
    while (!queue_out4_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return;
      }
    }
    std::string text =
        "abcd efgh ijkl mnop qrst uvwx abcd efgh ijkl mnop qrst uvwx abcd "
        "efgh ijkl mnop ";
    for (auto scale = 0.1; scale < 5; scale += 0.1) {
      int baseline = 0;
      cv::Size textSize =
          getTextSize(text, cv::FONT_HERSHEY_TRIPLEX, scale, 1, &baseline);
      if (textSize.width + 150 > layout4.dst_width) {
        base_height_ = layout4.dst_height / 3;
        text_height_ = textSize.height + (base_height_ - textSize.height) / 2;
        puttext_scale_ = scale;
        LOG(INFO) << "puttext : " << base_height_ << " " << text_height_ << " "
                  << puttext_scale_;
        break;
      }
    }

    if (ENV_PARAM(ENABLE_RNN)) {
      LOG(INFO) << "thread [" << name() << "] Py Initialize starts";
      Py_Initialize();
      CHECK(Py_IsInitialized())
          << "thread [" << name() << "] Py Initialize failed";

      PyRun_SimpleString("import sys");
      PyRun_SimpleString("sys.path.append('./')");

      pModule_ = PyImport_ImportModule("inference_mic");
      CHECK(pModule_) << "thread [" << name()
                      << "] Py import module inference_mic failed";
      pDict_ = PyModule_GetDict(pModule_);
      CHECK(pDict_) << "thread [" << name() << "] Py get dict failed";

      pClassMicProcess_ = PyDict_GetItemString(pDict_, "MicProcess");
      CHECK(pClassMicProcess_)
          << "thread [" << name() << "] Py fails to get MicProcess class";

      auto args = Py_BuildValue("(s)", wavfile.c_str());
      pInstanceMicProcess_ = PyEval_CallObject(pClassMicProcess_, args);
      CHECK(pClassMicProcess_)
          << "thread [" << name()
          << "] Py instantiation of MicProcess class fails";
      MicRun_ = PyObject_GetAttrString(pInstanceMicProcess_, "run");
      CHECK(MicRun_) << "thread [" << name()
                     << "] Py instantiation of MicProcess run function failed";

      LOG(INFO) << "thread [" << name() << "] Py Initialize ends";
    }
    auto video_stream = cv::VideoCapture(videofile);
    if (!video_stream.isOpened()) {
      LOG(ERROR) << "cannot open file " << videofile;
      stop();
    }
    cv::Mat image;
    while (video_stream.read(image)) {
      video_file_images_.push_back(image.clone());
    }
    CHECK_GT(video_file_images_.size(), 0);
    auto& layout1 = gui_layout()[1];
    layout1.src_width = video_file_images_[0].cols;
    layout1.src_height = video_file_images_[0].rows;
    gui_thread1_ = std::make_unique<vitis::ai::GuiThread>(1);
    queue_out1_ = gui_thread1_->getQueue();
    video_file_images_idx_ = 0;
  }

  virtual ~CNNThread() {
    if (ENV_PARAM(ENABLE_RNN)) {
      Py_DECREF(MicRun_);
      Py_DECREF(pInstanceMicProcess_);
      Py_DECREF(pClassMicProcess_);
      Py_DECREF(pDict_);
      Py_DECREF(pModule_);
      Py_Finalize();
      LOG(INFO) << "thread [" << name() << "] Py Finalize succeeded";
    }
  }

  void run_mic(int idx, bool last_one) {
    auto args = Py_BuildValue("(i)", idx);
    mtx_.lock();
    PyObject* pRet = PyObject_CallObject(MicRun_, args);
    mtx_.unlock();
    CHECK(pRet) << "thread [" << name()
                << "] Failed to call MicProcess run function";
    char* s;
    PyArg_Parse(pRet, "s", &s);
    all_text_ += std::string(s);
    int line = 80;  // 105

    while (3 * line < all_text_.size()) {
      all_text_.erase(0, line);
    }
    vitis::ai::FrameInfo frame;
    frame.mat = background_.clone();

    for (size_t i = 0; i < 3 && line * i < all_text_.size(); i++) {
      cv::putText(frame.mat, all_text_.substr(line * i, line),
                  cv::Point(10, base_height_ * i + text_height_),
                  cv::FONT_HERSHEY_TRIPLEX, puttext_scale_,
                  cv::Scalar(20, 20, 180), 1, 1);
    }

    if (last_one) all_text_ = "";
    while (!queue_out4_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return;
      }
    }
  }

  virtual int run() override {
    if (is_stopped()) {
      return -1;
    }
    vitis::ai::FrameInfo frame;
    frame.mat = video_file_images_[video_file_images_idx_++].clone();
    while (!queue_out1_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    if (video_file_images_idx_ % 179 == 178) {
      if (ENV_PARAM(ENABLE_RNN)) {
        micthread_ = std::thread(&CNNThread::run_mic, this,
                                 video_file_images_idx_ / 179, false);
        if (micthread_.joinable()) micthread_.detach();
      } else {
        vitis::ai::FrameInfo frame;
        frame.mat = background_.clone();
        while (!queue_out4_->push(frame, std::chrono::milliseconds(500))) {
          if (is_stopped()) {
            return -1;
          }
        }
      }
    }
    if (video_file_images_idx_ >= video_file_images_.size()) {
      if (ENV_PARAM(ENABLE_RNN)) {
        micthread_ = std::thread(&CNNThread::run_mic, this,
                                 video_file_images_idx_ / 179, true);
        if (micthread_.joinable()) micthread_.detach();
      }
      video_file_images_idx_ = 0;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(33));

    return 0;
  }

  virtual std::string name() override { return name_; }

  vitis::ai::queue_t* queue_out4_;
  vitis::ai::queue_t* queue_out1_;
  std::string all_text_;
  std::string name_;
  std::unique_ptr<vitis::ai::GuiThread> gui_thread1_;
  std::unique_ptr<vitis::ai::GuiThread> gui_thread4_;
  cv::Mat background_;
  PyObject* pModule_;
  PyObject* pDict_;
  PyObject* pClassMicProcess_;
  PyObject* pInstanceMicProcess_;
  PyObject* MicRun_;
  std::vector<cv::Mat> video_file_images_;
  int video_file_images_idx_;
  std::thread micthread_;
  std::mutex mtx_;
  float puttext_scale_;
  int text_height_;
  int base_height_;
};

int main(int argc, char* argv[]) {
  auto width = 3840;   // 1920;
  auto height = 2160;  // 1080;
  {
    int fd;
    struct fb_var_screeninfo screen_info;
    fd = open("/dev/fb0", O_RDWR);
    ioctl(fd, FBIOGET_VSCREENINFO, &screen_info);
    LOG(INFO) << screen_info.xres << " " << screen_info.yres;
    width = screen_info.xres;
    height = screen_info.yres;
    close(fd);
  }
  gui_layout() = {{0, 0, width / 2, height / 2},
                  {width / 2, 0, width / 2, 8 * height / 18},
                  {0, height / 2, width / 2, height / 2},
                  {width / 2, height / 2, width / 2, height / 2},
                  {width / 2, 8 * height / 18, width / 2, height / 18,
                   width / 2, height / 18}};

  gui_background() = cv::imread("/usr/share/weston/logo.jpg");
  // init each dpu filter and process instance, using video demo framework
  signal(SIGINT, vitis::ai::MyThread::signal_handler);
  vitis::ai::parse_opt(argc, argv);
  std::vector<vitis::ai::Channel> channels;
  channels.reserve(4);
  channels.emplace_back(
      0, vitis::ai::g_avi_file[0 % vitis::ai::g_avi_file.size()],
      [] {
        return vitis::ai::create_dpu_filter(
            [] { return vitis::ai::MultiTask8UC3::create("multi_task"); },
            process_result_multitask);
      },
      vitis::ai::g_num_of_threads[0 % vitis::ai::g_num_of_threads.size()]);
  auto cnnthread =
      CNNThread(vitis::ai::g_avi_file[1 % vitis::ai::g_avi_file.size()],
                vitis::ai::g_avi_file[4 % vitis::ai::g_avi_file.size()]);
  channels.emplace_back(
      2, vitis::ai::g_avi_file[2 % vitis::ai::g_avi_file.size()],
      [] {
        return vitis::ai::create_dpu_filter(
            [] { return vitis::ai::SSDPoseDetect::create(); },
            process_result_pose_detect_with_ssd);
      },
      vitis::ai::g_num_of_threads[2 % vitis::ai::g_num_of_threads.size()]);
  channels.emplace_back(
      3, vitis::ai::g_avi_file[3 % vitis::ai::g_avi_file.size()],
      [] {
        return vitis::ai::create_dpu_filter(
            [] { return vitis::ai::FaceDetect::create("densebox_640_360"); },
            process_result_facedetect);
      },
      vitis::ai::g_num_of_threads[3 % vitis::ai::g_num_of_threads.size()]);
  vitis::ai::MyThread::start_all();
  vitis::ai::MyThread::wait_all();
  vitis::ai::MyThread::stop_all();
  vitis::ai::MyThread::wait_all();
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";

  return 0;
}
