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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
//#include <vitis/ai/demo2.hpp>
#include <future>
#include <mutex>
#include <vitis/ai/demo.hpp>
#include <vitis/ai/pointpainting.hpp>
#include <vitis/ai/profiling.hpp>
#include "./demo_utils.hpp"
#include "./load_resource.hpp"

using namespace vitis::ai;
using namespace vitis::ai::pointpillars_nus;
using namespace vitis::ai::pointpainting;
using std::string;
using std::vector;

DEF_ENV_PARAM(USE_SMALL_SET, "0");
DEF_ENV_PARAM(DEBUG_SINGLE_THREAD, "0");
DEF_ENV_PARAM(DEBUG_RESULT, "0");
DEF_ENV_PARAM(SHOW_SEQ, "0");

const int g_load_thread_num = 4;
int g_work_thread_num = 6;

const int SHOW_WIDTH = 900;
const int SHOW_HEIGHT = 900;
const float CAM_RATIO = 0.5;
const int CAM_SHOW_WIDTH = 576 * CAM_RATIO;
const int CAM_SHOW_HEIGHT = 320 * CAM_RATIO;
const int BEV_SHOW_SIZE = 400;

using pt_queue_t = vitis::ai::BoundedQueue<PtFrameInfo>;

static void LoadNames(std::string const& filename,
                      std::vector<std::string>& input_file_names) {
  input_file_names.clear();

  /*Check if path is a valid directory path. */
  FILE* fp = fopen(filename.c_str(), "r");
  if (NULL == fp) {
    fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);
  }

  char buffer[256] = {0};
  while (fgets(buffer, 256, fp) != NULL) {
    int n = strlen(buffer);
    buffer[n - 1] = '\0';
    std::string name = buffer;
    input_file_names.push_back(name);
  }

  fclose(fp);
}

struct ReadThread : public MyThread {
  ReadThread(int channel_id, const std::string& data_root,
             const std::string& file, pt_queue_t* queue,
             bool use_preload = false)
      : MyThread{},
        channel_id_{channel_id},
        seq_index_{0},
        data_root_{data_root},
        file_{file},
        frame_id_{0},
        queue_{queue},
        use_preload_(use_preload) {
    // read seq list
    LoadNames(file_, seq_list_);
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "names size:" << seq_list_.size();
  }

  virtual ~ReadThread() {}

  static void load_resource(const std::string& data_root,
                            const std::string& file,
                            bool multi_thread = false) {
    std::vector<std::string> seq_list;
    LoadNames(file, seq_list);
    if (!ENV_PARAM(USE_SMALL_SET)) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "resources too large:" << seq_list.size();
      int size = seq_list.size();
      size = std::min(size, 200);
      seq_list.resize(size);
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "resources resize to:" << seq_list.size();
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "names size:" << seq_list.size();
    pre_points_infos_.resize(seq_list.size());
    int thread_num = multi_thread ? g_load_thread_num : 1;
    // std::vector<std::future> futures(thread_num);
    std::vector<std::thread> ts(thread_num);
    int size = seq_list.size();
    int batch = size / thread_num;
    for (auto i = 0; i < thread_num; ++i) {
      int start = i * batch;
      int end = start + batch;
      if (i == thread_num - 1) {
        end = size;
      }

      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "read thread[" << i << "] start:" << start << ", end:" << end;
      ts[i] = std::thread(load, std::cref(data_root), std::cref(seq_list),
                          std::ref(pre_points_infos_), start, end);
      // futures[i] =
      //    std::async(std::launch::async,
      //               [=, &points_infos = points_infos_, &seq_list = seq_list]
      //               {
      //                 load(seq_list, points_infos, start, end);
      //               });
    }
    LOG(INFO) << "resource loading, will take a few seconds";
    for (auto i = 0; i < thread_num; ++i) {
      ts[i].join();
    }
  }

  static unsigned int get_frame_id() {
    std::lock_guard<std::mutex> lock(mutex_);
    return pre_frame_id_++;
  }

  virtual int run() override {
    // read
    PointsInfoV2 points_info;

    if (use_preload_ && pre_points_infos_.size()) {
      auto frame_id = get_frame_id();
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "Read use preload frame:" << frame_id;
      while (!queue_->push(
          PtFrameInfo{channel_id_, frame_id,
                      pre_points_infos_[frame_id % pre_points_infos_.size()]},
          std::chrono::milliseconds(500))) {
        if (is_stopped()) {
          return -1;
        }
      }
    } else {
      if (seq_index_ == seq_list_.size()) {
        seq_index_ = 0;
        return 0;
      }
      __TIC__(READ_INNO)
      read_inno_file_v2(data_root_, seq_list_[seq_index_++], points_info, 5,
                        points_info.sweep_infos, 16, points_info.cams);
      __TOC__(READ_INNO)

      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "read queue size " << queue_->size();
      while (!queue_->push(PtFrameInfo{channel_id_, ++frame_id_, points_info},
                           std::chrono::milliseconds(500))) {
        if (is_stopped()) {
          return -1;
        }
      }
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "frame id:" << frame_id_ - 1 << " sended";
    }
    return 0;
  }

  virtual std::string name() override {
    return std::string{"ReadThread-"} + std::to_string(channel_id_);
  }

  static unsigned long pre_frame_id_;
  static vector<PointsInfoV2> pre_points_infos_;
  // vector<PointsInfoV2> points_infos_;
  int channel_id_;
  std::vector<std::string> seq_list_;
  uint32_t seq_index_;
  std::string data_root_;
  std::string file_;
  unsigned long frame_id_;
  pt_queue_t* queue_;
  bool use_preload_;
  static std::mutex mutex_;
};

unsigned long ReadThread::pre_frame_id_ = 0u;
vector<PointsInfoV2> ReadThread::pre_points_infos_ = vector<PointsInfoV2>();
std::mutex ReadThread::mutex_;

static cv::Mat merge_cam_bev(const std::vector<cv::Mat>& cams,
                             const cv::Mat& bev, unsigned long seq) {
  cv::Mat image(SHOW_WIDTH, SHOW_HEIGHT, CV_8UC3, cv::Scalar(0, 0, 0));
  // cv::Mat image(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0));
  // cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
      << "merge image shape:" << image.rows << " * " << image.cols << " * "
      << image.channels();
  assert(cams.size() >= 6);
  float ratio = 0.5;
  int w = CAM_SHOW_WIDTH;
  int h = CAM_SHOW_HEIGHT;

  // cams order:
  /// CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT,
  /// CAM_BACK_RIGHT

  //
  // display order:
  /// CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT
  /// CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT
  vector<int> indices{1, 2, 0, 4, 3, 5};
  for (auto i = 0; i < 6; ++i) {
    // i => indices[i]
    cv::Mat image_resize;
    if (cams[i].cols != w || cams[i].rows != h) {
      cv::resize(cams[i], image_resize, cv::Size{w, h});
    } else {
      image_resize = cams[i];
    }
    auto rect = cv::Mat(
        image, cv::Rect(w * (indices[i] % 3), h * (indices[i] / 3), w, h));
    // LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
    //    << "merge cams[" << i << "]"
    //    << " shape:" << cams[i].rows << " * " << cams[i].cols << " * "
    //    << cams[i].channels();

    // LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
    //    << "merge cams[" << i << "] rect"
    //    << " shape:" << rect.rows << " * " << rect.cols << " * "
    //    << rect.channels();

    // cams[i].copyTo(rect);
    image_resize.copyTo(rect);
  }
  cv::Mat bev_resize;
  if (bev.cols != BEV_SHOW_SIZE) {
    cv::resize(bev, bev_resize, cv::Size{BEV_SHOW_SIZE, BEV_SHOW_SIZE});
  } else {
    bev_resize = bev;
  }
  auto rect = cv::Mat(image, cv::Rect((image.cols - bev_resize.cols) / 2, h * 2,
                                      bev_resize.cols, bev_resize.rows));
  if (ENV_PARAM(SHOW_SEQ)) {
    cv::putText(bev_resize, std::string("seq:") + std::to_string(seq),
                cv::Point(7, 11), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 0, 0), 1, 1);
  }
  bev_resize.copyTo(rect);
  return image;
}

static std::vector<cv::Mat> draw_cams(const PointPaintingResult& result,
                                      const PtFrameInfo& frame,
                                      float resize_ratio = 0.5) {
  auto& cam_info = frame.pt_info.cam_info;
  std::vector<cv::Mat> cam_for_show(cam_info.size());
  for (auto cam_i = 0u; cam_i < cam_info.size(); ++cam_i) {
    auto s2l_t = cam_info[cam_i].s2l_t;
    auto s2l_r = cam_info[cam_i].s2l_r;
    auto cam_intr = cam_info[cam_i].cam_intr;
    auto l2c_mat = make_l2c_mat(s2l_r, s2l_t);
    auto intr = make_intrinsic_mat(cam_intr);
    auto l2c_v = mat2vector(l2c_mat);
    auto intr_v = mat2vector(intr);

    // frame.pt_info.cams[cam_i].copyTo(cam_for_show[cam_i]);
    int width = frame.pt_info.cams[cam_i].cols * resize_ratio;
    int height = frame.pt_info.cams[cam_i].rows * resize_ratio;
    cv::resize(frame.pt_info.cams[cam_i], cam_for_show[cam_i],
               cv::Size{width, height});
    for (auto i = 0u; i < result.bboxes.size(); ++i) {
      if (ENV_PARAM(DEBUG_RESULT)) {
        std::cout << "label: " << result.bboxes[i].label;
        std::cout << " bbox: ";
        for (auto j = 0u; j < result.bboxes[i].bbox.size(); ++j) {
          std::cout << result.bboxes[i].bbox[j] << " ";
        }
        std::cout << "score: " << result.bboxes[i].score;
        std::cout << std::endl;
      }
      auto& bbox = result.bboxes[i].bbox;

      vector<float> trans_bbox = vector<float>(bbox.begin(), bbox.begin() + 7);
      // draw_proj_box(trans_bbox, frame.pt_info.cams[cam_i], l2c_v, intr_v,
      draw_proj_box(trans_bbox, cam_for_show[cam_i], l2c_v, intr_v,
                    result.bboxes[i].label);
    }
  }
  return cam_for_show;
}

static cv::Mat draw_bev(const PointPaintingResult& result,
                        const PtFrameInfo& frame) {
  cv::Mat bev_for_show;
  // frame.pt_info.bev.copyTo(bev_for_show);
  cv::resize(frame.pt_info.bev, bev_for_show,
             cv::Size{BEV_SHOW_SIZE, BEV_SHOW_SIZE});

  std::vector<int> point_range{-50, -50, -5, 50, 50, 3};
  // float resolution = 0.125 / frame.pt_info.bev.cols * 800;
  float resolution = 100.f / BEV_SHOW_SIZE;
  for (auto i = 0u; i < result.bboxes.size(); ++i) {
    auto bev = make_bv_feature(point_range, result.bboxes[i].bbox, resolution,
                               bev_for_show.cols, bev_for_show.rows);
    // frame.pt_info.bev.cols, frame.pt_info.bev.rows);
    for (auto j = 0u; j < bev.size(); ++j) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_RESULT))
          << "bev[" << i << "]: line [" << j << "]:" << bev[j][0] << " "
          << bev[j][1] << " " << bev[j][2] << " " << bev[j][3];

      // cv::line(frame.pt_info.bev, cv::Point(bev[j][0], bev[j][1]),
      cv::line(bev_for_show, cv::Point(bev[j][0], bev[j][1]),
               cv::Point(bev[j][2], bev[j][3]), vscalar[result.bboxes[i].label],
               1);
      // cv::Point(bev[j][2], bev[j][3]), 0xff, 1);
    }
  }

  // try flip
  cv::flip(bev_for_show, bev_for_show, 1);
  return bev_for_show;
}

//
// Implement working thread
struct PtThread : public MyThread {
  PtThread(const std::string& seg_model, const std::string& model_0,
           const std::string& model_1, pt_queue_t* queue_in, queue_t* queue_out,
           const std::string& suffix)
      : MyThread{},
        seg_model_{seg_model},
        model_0_{model_0},
        model_1_{model_1},
        queue_in_{queue_in},
        queue_out_{queue_out},
        frame_id_{0},
        suffix_{suffix} {
    detector_ =
        vitis::ai::PointPainting::create(seg_model_, model_0_, model_1_);
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT Pt work";
  }
  virtual ~PtThread() {}
  virtual int run() override {
    PtFrameInfo frame;
    if (!queue_in_->pop(frame, std::chrono::milliseconds(500))) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "Pt work no frame";
      return 0;
    }

    auto ret = detector_->run(frame.pt_info.cams, frame.pt_info);
    auto cams_for_show = draw_cams(ret, frame);
    auto bev_for_show = draw_bev(ret, frame);

    __TIC__(MERGE)
    // auto image = merge_cam_bev(frame.pt_info.cams, frame.pt_info.bev,
    // frame.frame_id);
    auto image = merge_cam_bev(cams_for_show, bev_for_show, frame.frame_id);
    __TOC__(MERGE)

    // FrameInfo out_frame{frame.channel_id, frame.frame_id, frame.pt_info.bev};
    FrameInfo out_frame{frame.channel_id, frame.frame_id, image};
    while (!queue_out_->push(out_frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string{"PT-"} + suffix_; }
  std::string seg_model_;
  std::string model_0_;
  std::string model_1_;
  std::unique_ptr<PointPainting> detector_;
  pt_queue_t* queue_in_;
  queue_t* queue_out_;
  unsigned long frame_id_;
  std::string suffix_;
};
//
// Implement sorting thread
struct TSortingThread : public MyThread {
  TSortingThread(queue_t* queue_in, queue_t* queue_out,
                 const std::string& suffix)
      : MyThread{},
        queue_in_{queue_in},
        queue_out_{queue_out},
        frame_id_{0},
        suffix_{suffix},
        fps_{0.0f},
        max_fps_{0.0f} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT TSORTING";
  }
  virtual ~TSortingThread() {}
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
    // if (!queue_in_->pop(frame, std::chrono::milliseconds(500))) {
    //  return 0;
    //}
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "TSorting pop frame:" << frame.frame_id;
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
      max_fps_ = std::max(max_fps_, fps_);
      frame.max_fps = max_fps_;
      if (frame.mat.cols > 200)
        cv::putText(frame.mat, std::string("FPS: ") + std::to_string(fps),
                    cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(20, 20, 180), 2, 1);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] "
        << "col:" << frame.mat.cols << ", row:" << frame.mat.rows
        << " frame id " << frame.frame_id << " sorting queue size "
        << queue_out_->size() << "   FPS: " << fps;
    points_.push_front(now);
    if (duration > 2000) {  // sliding window for 2 seconds.
      points_.pop_back();
    }
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override {
    return std::string{"TSORT-"} + suffix_;
  }
  queue_t* queue_in_;
  queue_t* queue_out_;
  unsigned long frame_id_;
  std::deque<std::chrono::time_point<std::chrono::steady_clock>> points_;
  std::string suffix_;
  float fps_;
  float max_fps_;
};

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cout << "usage:" << argv[0]
              << " [segmentation_model] [pointpillars_model0] "
                 "[pointpillars_model1] [resource_root_dir] [seq_list]"
              << std::endl;
    exit(0);
  }

  signal(SIGINT, MyThread::signal_handler);

  std::string seg_model = argv[1];
  std::string model_0 = argv[2];
  std::string model_1 = argv[3];

  vector<string> names;
  std::string root_dir = argv[4];
  std::string input_file_list = argv[5];
  auto channel_id = 0;

  ReadThread::load_resource(root_dir, input_file_list, true);
  auto decode_queue = std::unique_ptr<pt_queue_t>{new pt_queue_t{5}};
  auto read_thread = std::unique_ptr<ReadThread>(new ReadThread{
      channel_id, root_dir, input_file_list, decode_queue.get(), true});

  auto gui_thread = GuiThread::instance();
  auto gui_queue = gui_thread->getQueue();

  // auto thread_num = 1;
  if (ENV_PARAM(DEBUG_SINGLE_THREAD)) {
    g_work_thread_num = 1;
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
      << "work thread num:" << g_work_thread_num;
  auto sorting_queue =
      std::unique_ptr<queue_t>(new queue_t(5 * g_work_thread_num));
  auto sorting_thread = std::unique_ptr<TSortingThread>(
      new TSortingThread(sorting_queue.get(), gui_queue, std::to_string(0)));
  auto dpu_thread = std::vector<std::unique_ptr<PtThread>>{};
  for (int i = 0; i < g_work_thread_num; ++i) {
    dpu_thread.emplace_back(
        new PtThread(seg_model, model_0, model_1, decode_queue.get(),
                     sorting_queue.get(), std::to_string(i)));
    // decode_queue.get(), gui_queue, std::to_string(i)));
  }

  MyThread::start_all();
  MyThread::wait_all();
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";

  return 0;
}

