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
size_t batch_size = 1;
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

inline std::vector<cv::Size>& each_channel_mosaik_size() {
  static std::vector<cv::Size> msize;
  return msize;
}

namespace vitis {
namespace ai {
// Read a video without doing anything
struct VideoByPass {
 public:
  int run(const cv::Mat& input_image) { return 0; }
};

// Do nothing after after excuting
inline cv::Mat process_none(cv::Mat image, int fake_result, bool is_jpeg) {
  return image;
}

// A struct that can storage data and info for each frame
struct FrameInfo {
  int channel_id;
  unsigned long frame_id;
  cv::Mat mat;
  float max_fps;
  float fps;
  int belonging;
  int mosaik_width;
  int mosaik_height;
  int horizontal_num;
  int vertical_num;
  cv::Rect_<int> local_rect;
  cv::Rect_<int> page_layout;
  std::string channel_name;
  std::vector<cv::Mat> mats;
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

// std::vector<MyThread *> MyThread::all_threads_;
struct DecodeThread : public MyThread {
  DecodeThread(int channel_id, const std::string& video_file, queue_t* queue)
      : MyThread{},
        channel_id_{channel_id},
        video_file_{video_file},
        frame_id_{0},
        video_stream_{},
        queue_{queue} {
    is_camera_ = video_file_.size() == 1 && video_file_[0] >= '0' &&
                 video_file_[0] <= '9';
    auto& layout = gui_layout()[channel_id_];

    if (is_camera_) {
      video_stream_ = cv::VideoCapture(std::stoi(video_file_));
      if (!video_stream_.isOpened()) {
        LOG(ERROR) << "cannot open camera " << video_file_;
        exit(1);
      }
      video_stream_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      video_stream_.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
      layout.src_width = 640;
      layout.src_height = 360;
    } else {
      video_stream_ = cv::VideoCapture(video_file_);
      if (!video_stream_.isOpened()) {
        LOG(ERROR) << "cannot open file " << video_file_;
        exit(1);
      }
      cv::Mat image;
      while (video_stream_.read(image)) {
        video_file_images_.push_back(image.clone());
      }
      CHECK_GT(video_file_images_.size(), 0);
      layout.src_width = video_file_images_[0].cols;
      layout.src_height = video_file_images_[0].rows;
      video_file_images_idx_ = 0;
    }
  }

  virtual ~DecodeThread() { video_stream_.release(); }

  virtual int run() override {
    if (is_stopped()) {
      return -1;
    }
    FrameInfo frameinfo{channel_id_, ++frame_id_};
    for (size_t i = 0; i < batch_size; i++) {
      cv::Mat image;
      int j = 0;
      if (is_camera_) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_DEDODE)) << "begin read";
        while (!video_stream_.read(image)) {
          LOG(INFO) << "[" << name() << "] cannot read from camera " << j;
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          if (j++ > 30) return -1;
        }
        LOG_IF(INFO, ENV_PARAM(DEBUG_DEDODE)) << "end read";
      } else {
        image = video_file_images_[video_file_images_idx_++].clone();
        if (video_file_images_idx_ >= video_file_images_.size())
          video_file_images_idx_ = 0;
      }
      frameinfo.mats.push_back(image);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEDODE))
        << "[" << name() << "]decode queue size " << queue_->size();

    while (!queue_->push(frameinfo, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override {
    return std::string{"DedodeThread-"} + std::to_string(channel_id_);
  }

  int channel_id_;
  std::string video_file_;
  unsigned long frame_id_;
  cv::VideoCapture video_stream_;
  std::vector<cv::Mat> video_file_images_;
  int video_file_images_idx_;
  queue_t* queue_;
  bool is_camera_;
};

static std::unique_ptr<cv::VideoWriter> maybe_create_gst_video_writer(
    int x, int y, int width, int height, int src_width, int src_height,
    int id) {
  std::vector <int > scaler{1,2,4,5};
  std::string pipeline;
  if (id != 4) {
    pipeline =
        std::string("appsrc ! queue ! videoconvert ! queue ") +
        " ! vvas_xabrscaler "
        "xclbin-location=\"/media/sd-mmcblk0p1/dpu.xclbin\" " +
        "kernel-name=v_multi_scaler:v_multi_scaler_" + std::to_string(scaler[id]) +
        " ! video/x-raw, width=" + std::to_string(width) + "," +
        " height=" + std::to_string(height) +
        "! queue ! kmssink driver-name=xlnx plane-id=" +
        std::to_string(34 + id) + " render-rectangle=\"<" + std::to_string(x) +
        "," + std::to_string(y) + "," + std::to_string(width) + "," +
        std::to_string(height) + ">\" sync=false";
  } else {
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

    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "[" << name() << "] queue size " << queue_->size()
        << ", state = " << (is_stopped() ? "stopped" : "running")
        << " mat.size: " << frame_info.mat.size().height << " "
        << frame_info.mat.size().width;

    LOG_IF(INFO, ENV_PARAM(DEBUG_GUI) == id_) << name() << " push to pipeline";

    *video_writer_ << frame_info.mat;
    LOG_IF(INFO, ENV_PARAM(DEBUG_GUI) == id_)
        << name() << " push to pipeline end";
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

struct Filter {
  explicit Filter() {}
  virtual ~Filter() {}
  virtual std::vector<cv::Mat> run(std::vector<cv::Mat>& input) = 0;
};

// Execute each lib run function and processor your implement
template <typename dpu_model_type_t, typename ProcessResult>
struct DpuFilter : public Filter {
  DpuFilter(std::unique_ptr<dpu_model_type_t>&& dpu_model,
            const ProcessResult& processor)
      : Filter{}, dpu_model_{std::move(dpu_model)}, processor_{processor} {
    LOG(INFO) << "DPU model size=" << dpu_model_->getInputWidth() << "x"
              << dpu_model_->getInputHeight();
  }
  virtual ~DpuFilter() {}
  std::vector<cv::Mat> run(std::vector<cv::Mat>& images) override {
    auto results = dpu_model_->run(images);
    return processor_(images, results, false);
  }
  std::unique_ptr<dpu_model_type_t> dpu_model_;
  const ProcessResult& processor_;
};
template <typename FactoryMethod, typename ProcessResult>
std::unique_ptr<Filter> create_dpu_filter(const FactoryMethod& factory_method,
                                          const ProcessResult& process_result) {
  using dpu_model_type_t = typename decltype(factory_method())::element_type;
  return std::unique_ptr<Filter>(new DpuFilter<dpu_model_type_t, ProcessResult>(
      factory_method(), process_result));
}

// Execute dpu filter
struct DpuThread : public MyThread {
  DpuThread(std::unique_ptr<Filter>&& filter, queue_t* queue_in,
            queue_t* queue_out, const std::string& suffix)
      : MyThread{},
        filter_{std::move(filter)},
        queue_in_{queue_in},
        queue_out_{queue_out},
        suffix_{suffix} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT DPU";
  }
  virtual ~DpuThread() {}

  virtual int run() override {
    FrameInfo frame;
    if (!queue_in_->pop(frame, std::chrono::milliseconds(500))) {
      return 0;
    }
    if (filter_) {
      std::vector<cv::Mat> mats = frame.mats;
      frame.mats = filter_->run(mats);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "dpu queue size " << queue_out_->size();
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string("DPU-") + suffix_; }
  std::unique_ptr<Filter> filter_;
  queue_t* queue_in_;
  queue_t* queue_out_;
  std::string suffix_;
};

// Implement sorting thread
struct SortingThread : public MyThread {
  SortingThread(queue_t* queue_in, queue_t* queue_out,
                const std::string& suffix)
      : MyThread{},
        queue_in_{queue_in},
        queue_out_{queue_out},
        frame_id_{0},
        suffix_{suffix},
        fps_{0.0f},
        max_fps_{0.0f} {
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
      fps = fps * batch_size;
      auto x = 10;
      auto y = 20;
      fps_ = fps;
      frame.fps = fps;
      max_fps_ = std::max(max_fps_, fps_);
      frame.max_fps = max_fps_;
      // if (frame.mat.cols > 200)
      //   cv::putText(frame.mat, std::string("FPS: ") + std::to_string(fps),
      //               cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
      //               cv::Scalar(20, 20, 180), 2, 1);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] "
        << " frame id " << frame.frame_id << " sorting queue size "
        << queue_out_->size() << "   FPS: " << fps;
    points_.push_front(now);
    if (duration > 2000) {  // sliding window for 2 seconds.
      points_.pop_back();
    }
    for (auto i = 0; i < batch_size; i++) {
      auto s_frame = frame;
      s_frame.mat = frame.mats[i];
      if (s_frame.mat.cols > 200)
        cv::putText(s_frame.mat, std::string("FPS: ") + std::to_string(fps),
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(20, 20, 180), 2, 1);
      while (!queue_out_->push(s_frame, std::chrono::milliseconds(500))) {
        if (is_stopped()) {
          return -1;
        }
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string{"SORT-"} + suffix_; }
  queue_t* queue_in_;
  queue_t* queue_out_;
  unsigned long frame_id_;
  std::deque<std::chrono::time_point<std::chrono::steady_clock>> points_;
  std::string suffix_;
  float fps_;
  float max_fps_;
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

// A class can create a video channel
struct Channel {
  Channel(size_t ch, const std::string& avi_file,
          const std::function<std::unique_ptr<Filter>()>& filter,
          int n_of_threads) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "create channel " << ch << " for " << avi_file;
    auto channel_id = ch;
    decode_queue = std::unique_ptr<queue_t>(new queue_t(10));
    decode_thread = std::unique_ptr<DecodeThread>(
        new DecodeThread{(int)channel_id, avi_file, decode_queue.get()});
    dpu_thread = std::vector<std::unique_ptr<DpuThread>>{};
    sorting_queue = std::unique_ptr<queue_t>(new queue_t(5 * n_of_threads));
    for (int i = 0; i < n_of_threads; ++i) {
      auto suffix =
          avi_file + "-" + std::to_string(i) + "/" + std::to_string(ch);
      dpu_thread.emplace_back(new DpuThread{filter(), decode_queue.get(),
                                            sorting_queue.get(), suffix});
    }
    gui_thread = std::make_unique<GuiThread>(channel_id);
    auto gui_queue = gui_thread->getQueue();
    sorting_thread = std::unique_ptr<SortingThread>(new SortingThread(
        sorting_queue.get(), gui_queue, avi_file + "-" + std::to_string(ch)));
  }

  std::unique_ptr<queue_t> decode_queue;
  std::unique_ptr<DecodeThread> decode_thread;
  std::vector<std::unique_ptr<DpuThread>> dpu_thread;
  std::unique_ptr<queue_t> sorting_queue;
  std::unique_ptr<SortingThread> sorting_thread;
  std::unique_ptr<GuiThread> gui_thread;
};

}  // namespace ai
}  // namespace vitis
