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
#include <glog/logging.h>
#include <opencv2/imgproc/types_c.h>
#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>
#include <type_traits>
#include <vitis/ai/bounded_queue.hpp>
#include <vitis/ai/env_config.hpp>

#include "process_result.hpp"

DEF_ENV_PARAM(DEBUG_DEMO, "0")
DEF_ENV_PARAM(DEBUG_DEDODE, "0")
DEF_ENV_PARAM(DEBUG_GUI, "-1")
DEF_ENV_PARAM(DEMO_USE_VIDEO_WRITER, "0")
DEF_ENV_PARAM_2(
    DEMO_VIDEO_WRITER,
    "appsrc ! videoconvert ! queue ! fpsdisplaysink video-sink=\"kmssink "
    "driver-name=xlnx plane-id=39\" sync=false -v ",
    std::string)
DEF_ENV_PARAM(DEMO_VIDEO_WRITER_WIDTH, "640")
DEF_ENV_PARAM(DEMO_VIDEO_WRITER_HEIGHT, "480")
DEF_ENV_PARAM(SAMPLES_ENABLE_BATCH, "1");

// set the layout
struct MyRect {
  int x;
  int y;
  int dst_width;
  int dst_height;
  int src_width;
  int src_height;
};

inline std::vector<MyRect>& gui_layout() {
  static std::vector<MyRect> rects;
  return rects;
}
// set the wallpaper
inline cv::Mat& gui_background() {
  static cv::Mat img;
  return img;
}

namespace vitis {
namespace ai {

// A struct that can storage data and info for each frame
struct FrameInfo {
  unsigned long frame_id;
  std::vector<cv::Mat> resize_images;
  std::vector<std::vector<char>> bins;
  cv::Mat gui_image;
  cv::Mat mat;
  float fps;
};

using queue_t = vitis::ai::BoundedQueue<FrameInfo>;
struct MyThread {
  // static std::vector<MyThread *> all_threads_;
  static inline std::vector<MyThread*>& all_threads() {
    static std::vector<MyThread*> threads;
    return threads;
  };
  static void signal_handler(int) { stop_all(); }
  static void stop_all() {
    for (auto& th : all_threads()) {
      th->stop();
    }
  }
  static void wait_all() {
    for (auto& th : all_threads()) {
      th->wait();
    }
  }
  static void start_all() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "Thread num " << all_threads().size();
    for (auto& th : all_threads()) {
      th->start();
    }
  }

  static void main_proxy(MyThread* me) { return me->main(); }
  void main() {
    LOG(INFO) << "thread [" << name() << "] is started";
    while (!stop_) {
      auto run_ret = run();
      if (!stop_) {
        stop_ = run_ret != 0;
      }
    }
    LOG(INFO) << "thread [" << name() << "] is ended";
  }

  virtual int run() = 0;

  virtual std::string name() = 0;

  explicit MyThread() : stop_(false), thread_{nullptr} {
    // LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT A Thread";
    all_threads().push_back(this);
  }

  virtual ~MyThread() {  //
    all_threads().erase(
        std::remove(all_threads().begin(), all_threads().end(), this),
        all_threads().end());
  }

  void start() {
    thread_ = std::unique_ptr<std::thread>(new std::thread(main_proxy, this));
  }

  void stop() {
    LOG(INFO) << "thread [" << name() << "] is stopped.";
    stop_ = true;
  }

  void wait() {
    if (thread_ && thread_->joinable()) {
      LOG(INFO) << "waiting for [" << name() << "] ended";
      thread_->join();
    }
  }
  bool is_stopped() { return stop_; }

  bool stop_;
  std::unique_ptr<std::thread> thread_;
};

static std::unique_ptr<cv::VideoWriter> maybe_create_gst_video_writer(
    int x, int y, int width, int height, int src_width, int src_height,
    int id) {
  std::vector<int> scaler{1, 2, 4, 5};
  std::string pipeline;
  // if (id != 4) {
  //   pipeline = std::string("appsrc ! queue ! videoconvert ! queue ") +
  //              " ! vvas_xabrscaler "
  //              "xclbin-location=\"/media/sd-mmcblk0p1/dpu.xclbin\" " +
  //              "kernel-name=v_multi_scaler:v_multi_scaler_" +
  //              std::to_string(scaler[id]) +
  //              " ! video/x-raw, width=" + std::to_string(width) + "," +
  //              " height=" + std::to_string(height) +
  //              "! queue ! kmssink driver-name=xlnx plane-id=" +
  //              std::to_string(34 + id) + " render-rectangle=\"<" +
  //              std::to_string(x) + "," + std::to_string(y) + "," +
  //              std::to_string(width) + "," + std::to_string(height) +
  //              ">\" sync=false";
  // } else
  {
    pipeline = std::string("appsrc ! videoconvert ") +
               " ! queue ! kmssink driver-name=xlnx plane-id=" +
               std::to_string(34 + id) + " render-rectangle=\"<" +
               std::to_string(x) + "," + std::to_string(y) + "," +
               std::to_string(width) + "," + std::to_string(height) +
               ">\" sync=false";
  }
  auto video_stream = std::unique_ptr<cv::VideoWriter>(
      new cv::VideoWriter(pipeline, cv::CAP_GSTREAMER, 0, 25.0,
                          cv::Size(src_width, src_height), true));
  auto& writer = *video_stream.get();
  if (!writer.isOpened()) {
    LOG(FATAL) << "cannot open gst: " << pipeline;
    return nullptr;
  } else {
    LOG(INFO) << "video writer is created: " << src_width << "x" << src_height
              << " " << pipeline;
  }
  return video_stream;
}
static std::unique_ptr<cv::VideoWriter> maybe_create_gst_video_writer(int id) {
  auto layout = gui_layout()[id];
  return maybe_create_gst_video_writer(layout.x, layout.y, layout.dst_width,
                                       layout.dst_height, layout.src_width,
                                       layout.src_height, id);
}
struct GuiThread : public MyThread {
  // assuming GUI is not bottleneck, 10 is high enough
  GuiThread(int id)
      : MyThread{},
        queue_{new queue_t{10}},
        id_(id),
        name_{std::string("GUIThread-") + std::to_string(id)} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT GUI";
    layout_ = gui_layout()[id];
    video_writer_ = maybe_create_gst_video_writer(
        layout_.x, layout_.y, layout_.dst_width, layout_.dst_height,
        layout_.src_width, layout_.src_height, id);
  }
  virtual ~GuiThread() {}

  virtual int run() override {
    if (is_stopped()) {
      return -1;
    }
    FrameInfo frame_info;
    if (!queue_->pop(frame_info, std::chrono::milliseconds(500))) {
      return 0;
    }

    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO) > 2)
        << "[" << name() << "] queue size " << queue_->size()
        << ", state = " << (is_stopped() ? "stopped" : "running")
        << " mat.size: " << frame_info.mat.size().height << " "
        << frame_info.mat.size().width;

    *video_writer_ << frame_info.mat;
    return 0;
  }

  virtual std::string name() override { return name_; }

  queue_t* getQueue() { return queue_.get(); }

  std::unique_ptr<queue_t> queue_;
  std::unique_ptr<cv::VideoWriter> video_writer_;
  std::string name_;
  MyRect layout_;
  int id_;
};

// Implement sorting thread
struct SortingThread : public MyThread {
  SortingThread(queue_t* queue_in, queue_t* queue_out[],
                const std::string& suffix)
      : MyThread{},
        queue_in_{queue_in},
        frame_id_{0},
        suffix_{suffix},
        fps_{0.0f} {
    for (size_t i = 0; i < 2; i++) {
      queue_out_[i] = queue_out[i];
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT SORTING";
  }
  virtual ~SortingThread() {}
  virtual int run() override {
    FrameInfo frame;
    frame_id_++;
    auto frame_id = frame_id_;
    auto cond =
        std::function<bool(const FrameInfo&)>{[frame_id](const FrameInfo& f) {
          // sorted by frame id
          return f.frame_id <= frame_id;
        }};
    if (!queue_in_->pop(frame, cond, std::chrono::milliseconds(500))) {
      return 0;
    }
    auto now = std::chrono::steady_clock::now();
    float fps = -1.0f;
    long duration = 0;
    if (!points_.empty()) {
      auto end = points_.back();
      duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - end)
              .count();
      float duration2 = (float)duration;
      float total = (float)points_.size();
      fps = total / duration2 * 1000.0f;
      auto x = 10;
      auto y = 20;
      fps_ = fps;
      frame.fps = fps;
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] "
        << " frame id " << frame.frame_id << " sorting queue size "
        << queue_out_[1]->size() << "   FPS: " << fps;
    points_.push_front(now);
    if (duration > 2000) {  // sliding window for 2 seconds.
      points_.pop_back();
    }
    if (frame.mat.cols > 200)
      cv::putText(frame.mat, std::string("FPS: ") + std::to_string(fps),
                  cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(20, 20, 180), 2, 1);
    while (!queue_out_[1]->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) return -1;
    }
    FrameInfo frame_to_gui;
    frame_to_gui.mat = frame.gui_image;
    while (!queue_out_[0]->push(frame_to_gui, std::chrono::milliseconds(500))) {
      if (is_stopped()) return -1;
    }
    return 0;
  }

  virtual std::string name() override { return std::string{"SORT-"} + suffix_; }
  queue_t* queue_in_;
  // queue_t* queue_out_;
  queue_t* queue_out_[2];
  unsigned long frame_id_;
  std::deque<std::chrono::time_point<std::chrono::steady_clock>> points_;
  std::string suffix_;
  float fps_;
};

static std::vector<std::string> LoadListNames(const std::string& filename) {
  std::vector<std::string> vlist;
  std::ifstream Tin(filename);
  std::string str;
  if (!Tin) {
    std::cout << "Can't open the file " << filename << "\n";
    exit(-1);
  }
  while (getline(Tin, str)) {
    vlist.emplace_back(str);
  }
  Tin.close();
  return vlist;
}

cv::Mat resize_and_crop_image(const cv::Mat& image) {
  auto original_height = 900, original_width = 1600;
  auto final_height = 256, final_width = 704;
  auto resize_scale = 0.48;  //

  int resized_width = original_width * resize_scale;
  int resized_height = original_height * resize_scale;
  cv::Size resize_dims(resized_width, resized_height);
  auto crop_h = std::max(0, resized_height - final_height);
  auto crop_w = int(std::max(0, (resized_width - final_width) / 2));
  cv::Rect box(crop_w, crop_h, final_width, final_height);

  if (image.size() == resize_dims) {
    return image(box).clone();
  }
  if (image.size() == cv::Size(final_width, final_height)) {
    return image;
  }
  cv::Mat img;
  cv::resize(image, img, resize_dims, 0, 0, cv::INTER_LINEAR);
  auto res = img(box).clone();
  return res;
}

struct DecodeThread : public MyThread {
  DecodeThread(const std::string& data_name_file, const std::string& data_path,
               queue_t* queue)
      : MyThread{}, frame_id_{0}, queue_{queue} {
    auto names = LoadListNames(data_name_file);

    // pic: _cam0.jpg, ..._cam5.jpg
    // bins: translation rotation points_z  points
    std::vector<std::string> bin_name_ext{"translation", "rotation", "points_z",
                                          "points"};
    images_.resize(names.size());
    resize_images_.resize(names.size());
    bins_.resize(names.size());
    gui_images_.resize(names.size());
    for (size_t i = 0; i < names.size(); i++) {
      std::string name_base = data_path + "/" + names[i] + "/";
      images_[i].resize(0);
      resize_images_[i].resize(0);
      bins_[i].resize(0);
      cv::Size dst_size(gui_layout()[0].src_width / 3,
                        gui_layout()[0].src_height / 2);
      std::vector<cv::Mat> line0(3), line1(3), concat(2);
      std::vector<int> idx_to_dstidx{1, 2, 0, 4, 3, 5};
      for (int j = 0; j < 6; j++) {
        auto pic_name = name_base + "cam" + std::to_string(j) + ".jpg";
        images_[i].emplace_back(cv::imread(pic_name));
        resize_images_[i].emplace_back(resize_and_crop_image(images_[i][j]));
        if (idx_to_dstidx[j] < 3)
          cv::resize(images_[i][j], line0[idx_to_dstidx[j]], dst_size);
        else
          cv::resize(images_[i][j], line1[idx_to_dstidx[j] - 3], dst_size);
      }
      cv::hconcat(line0, concat[0]);
      cv::hconcat(line1, concat[1]);
      cv::vconcat(concat, gui_images_[i]);
      if (ENV_PARAM(DEBUG_DEDODE)) {
        cv::putText(gui_images_[i], std::string("ID: ") + names[i],
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(20, 20, 180), 2, 1);
      }
      for (int j = 0; j < 4; j++) {
        auto bin_name = name_base + bin_name_ext[j] + ".bin";
        auto infile = std::ifstream(bin_name, std::ios_base::binary);
        bins_[i].emplace_back(
            std::vector<char>(std::istreambuf_iterator<char>(infile),
                              std::istreambuf_iterator<char>()));
      }
    }

    CHECK_GT(names.size(), 0);
    video_file_images_idx_ = 0;
  }

  virtual ~DecodeThread() {}

  virtual int run() override {
    if (is_stopped()) {
      return -1;
    }
    FrameInfo frameinfo{++frame_id_, resize_images_[video_file_images_idx_],
                        bins_[video_file_images_idx_],
                        gui_images_[video_file_images_idx_]};

    video_file_images_idx_++;
    if (video_file_images_idx_ >= images_.size()) video_file_images_idx_ = 0;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEDODE))
        << "[" << name() << "]decode queue size " << queue_->size();
    while (!queue_->push(frameinfo, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string{"DedodeThread"}; }

  unsigned long frame_id_;
  std::vector<std::vector<cv::Mat>> images_;
  std::vector<cv::Mat> gui_images_;
  std::vector<std::vector<cv::Mat>> resize_images_;
  std::vector<std::vector<std::vector<char>>> bins_;
  int video_file_images_idx_;
  queue_t* queue_;
};

struct BevdetThread : public MyThread {
  BevdetThread(queue_t* dpuin, queue_t* dpuout)
      : MyThread{},
        name_{std::string("BevdetThread")},
        dpuin_(dpuin),
        dpuout_(dpuout),
        bevdet_runner_{
            BEVdet::create("bevdet_0_pt", "bevdet_1_pt", "bevdet_2_pt")} {}

  virtual ~BevdetThread() {}

  virtual int run() override {
    if (is_stopped()) {
      return -1;
    }
    FrameInfo frame;
    if (!dpuin_->pop(frame, std::chrono::milliseconds(500))) {
      return 0;
    }
    auto res = bevdet_runner_->run(frame.resize_images, frame.bins);
    frame.mat = draw_bev(res, float(gui_layout()[1].dst_width) / 128.0f);
    while (!dpuout_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) return -1;
    }
    return 0;
  }

  virtual std::string name() override { return name_; }

  std::string name_;
  queue_t* dpuin_;
  queue_t* dpuout_;
  std::unique_ptr<BEVdet> bevdet_runner_;
};

inline void usage_video(const char* progname) {
  std::cout << "usage: " << progname << "      -t <num_of_threads>\n"
            << "      <video file name>\n"
            << std::endl;
  return;
}
/*
  global command line options
 */
static std::vector<int> g_num_of_threads;
static std::vector<std::string> g_avi_file;

inline void parse_opt(int argc, char* argv[], int start_pos = 1) {
  int opt = 0;
  optind = start_pos;
  while ((opt = getopt(argc, argv, "c:t:")) != -1) {
    switch (opt) {
      case 't':
        g_num_of_threads.emplace_back(std::stoi(optarg));
        break;
      case 'c':  // how many channels
        break;   // do nothing. parse it in outside logic.
      default:
        usage_video(argv[0]);
        exit(1);
    }
  }
  for (int i = optind; i < argc; ++i) {
    g_avi_file.push_back(std::string(argv[i]));
  }
  if (g_avi_file.empty()) {
    std::cerr << "Expected argument after options\n";
    exit(EXIT_FAILURE);
  }
  if (g_num_of_threads.empty()) {
    // by default, all channels has at least one thread
    g_num_of_threads.emplace_back(1);
  }
  return;
}

}  // namespace ai
}  // namespace vitis
