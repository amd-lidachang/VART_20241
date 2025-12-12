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

#include <fcntl.h>
#include <glog/logging.h>
#include <linux/fb.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/bevdet.hpp>
#include <vitis/ai/profiling.hpp>

#include "demo.hpp"
using namespace std;
using namespace cv;

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
  gui_layout() = {
      {0, height *2/ 3, width , height / 3, width , height / 3},
      {width / 2 - height / 3, 0, height *2/ 3, height *2/ 3, height *2/ 3, height *2/ 3}};
      //{width / 2 - height / 3, 0, height *2/ 3, height *2/ 3, 512, 512}}; //need use hard resize

  gui_background() = cv::imread("/usr/share/weston/logo.jpg");
  // init each dpu filter and process instance, using video demo framework
  signal(SIGINT, vitis::ai::MyThread::signal_handler);
  // vitis::ai::parse_opt(argc, argv);
  LOG(INFO) << argv[1];
  vitis::ai::queue_t dpuin(10), dpuout(10);
  auto decode = vitis::ai::DecodeThread(argv[1], argv[2], &dpuin);
  std::vector<std::unique_ptr<vitis::ai::GuiThread>> gui_thread;
  std::vector<vitis::ai::queue_t*> gui_in(gui_layout().size());
  for (int j = 0; j < gui_layout().size(); j++) {
    gui_thread.emplace_back(
        std::move(std::make_unique<vitis::ai::GuiThread>(j)));
    gui_in[j] = gui_thread[j]->getQueue();
  }
  auto sorting_thread = vitis::ai::SortingThread(
      &dpuout, gui_in.data(), std::string("test") + "-" + std::to_string(1));

  auto cnnthread0 = vitis::ai::BevdetThread(&dpuin, &dpuout);
  auto cnnthread1 = vitis::ai::BevdetThread(&dpuin, &dpuout);
  auto cnnthread2 = vitis::ai::BevdetThread(&dpuin, &dpuout);
  auto cnnthread3 = vitis::ai::BevdetThread(&dpuin, &dpuout);

  vitis::ai::MyThread::start_all();
  vitis::ai::MyThread::wait_all();
  vitis::ai::MyThread::stop_all();
  vitis::ai::MyThread::wait_all();
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";

  return 0;
}
