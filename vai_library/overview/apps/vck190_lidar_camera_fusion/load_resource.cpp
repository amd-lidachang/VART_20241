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

#include <sys/stat.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/profiling.hpp>
#include "./load_resource.hpp"


using namespace vitis::ai::pointpillars_nus;
using std::string;
using std::vector;

namespace vitis {
namespace ai {
namespace pointpainting {

DEF_ENV_PARAM(DEBUG_READ, "0");
DEF_ENV_PARAM(DEBUG_BEV_RESIZE, "0");

bool g_use_png = true;  // use resized cam image

void load(const std::string& data_root, const std::vector<std::string>& seq_list,
          std::vector<PointsInfoV2>& points_infos, int start, int end) {
  for (auto i = start; i < end; ++i) {
    read_inno_file_v2(data_root, seq_list[i], points_infos[i], 5,
                      points_infos[i].sweep_infos, 16, points_infos[i].cams);
    LOG_IF(INFO, ENV_PARAM(DEBUG_READ))
        << "read[" << i << "] :" << seq_list[i] << " done";
  }
}


static void read_cam_intr_from_line(const std::string& line,
                                    CamInfo& cam_info) {
  auto s = line;
  auto cnt = 0u;
  while (cnt < cam_info.cam_intr.size()) {
    auto f = std::atof(s.c_str());
    cam_info.cam_intr[cnt] = f;
    cnt++;
    auto n = s.find_first_of(' ');
    if (n == std::string::npos) {
      break;
    }
    s = s.substr(n + 1);
  }
}

static void read_s2lr_from_line(const std::string& line, CamInfo& cam_info) {
  auto s = line;
  auto cnt = 0u;
  while (cnt < cam_info.s2l_r.size()) {
    auto f = std::atof(s.c_str());
    cam_info.s2l_r[cnt] = f;
    cnt++;
    auto n = s.find_first_of(' ');
    if (n == std::string::npos) {
      break;
    }
    s = s.substr(n + 1);
  }
}

static void read_s2lt_from_line(const std::string& line, CamInfo& cam_info) {
  auto s = line;
  auto cnt = 0u;
  while (cnt < cam_info.s2l_t.size()) {
    auto f = std::atof(s.c_str());
    cam_info.s2l_t[cnt] = f;
    cnt++;
    auto n = s.find_first_of(' ');
    if (n == std::string::npos) {
      break;
    }
    s = s.substr(n + 1);
  }
}


static void read_points_file(const std::string& points_file_name,
                             std::vector<float>& points) {
  // int DIM = 5;
  // points_info.dim = DIM;
  struct stat file_stat;
  if (stat(points_file_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << points_file_name << " state error!" << std::endl;
    exit(-1);
  }
  auto file_size = file_stat.st_size;
  // LOG(INFO) << "input file:" << points_file_name << " size:" << file_size;
  // points_info.points.resize(file_size / 4);
  points.resize(file_size / 4);
  // CHECK(std::ifstream(points_file_name).read(reinterpret_cast<char
  // *>(points_info.points.data()), file_size).good());
  CHECK(std::ifstream(points_file_name)
            .read(reinterpret_cast<char*>(points.data()), file_size)
            .good());
}

static void read_sweeps(std::ifstream& anno_file,
                        const std::string& path_prefix,
                        std::vector<SweepInfo>& sweeps, int points_dim,
                        int space = 2) {
  char line[1024];
  std::string v_line;  // valid line
  // read sweeps
  if (anno_file.getline(line, 1024, '\n') &&
      std::strncmp(line, "sweeps:", 7) == 0) {
    auto s = std::string(line + 7);
    int num = std::atoi(s.c_str());
    // std::cout << "sweep num:" << num << std::endl;
    LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "sweep num:" << num;
    if (num == 0) {
      return;
    }
    int cnt = 0;
    sweeps.clear();
    sweeps.resize(num);
    while (cnt < num) {
      anno_file.getline(line, 1024, '\n');
      v_line = line + space;

      // read data path
      if (std::strncmp(v_line.c_str(), "data_path:", 10) == 0) {
        auto data_path = std::string(v_line).substr(10);
        data_path = path_prefix + data_path;
        LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "sweep data_path:" << data_path;
        // std::cout << "data_path:" << data_path << std::endl;
        if (!sweeps[cnt].points.points) {
          sweeps[cnt].points.points.reset(new std::vector<float>);
        }
        read_points_file(data_path, *(sweeps[cnt].points.points));
        sweeps[cnt].points.dim = points_dim;
      } else {
        break;
      }

      anno_file.getline(line, 1024, '\n');
      v_line = line + space;
      if (std::strncmp(v_line.c_str(), "timestamp:", 10) == 0) {
        auto timestamp = std::atoll(std::string(v_line).substr(10).c_str());
        LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "sweep timestamp:" << timestamp;
        // std::cout << "timestamp:" << timestamp<< std::endl;
        sweeps[cnt].cam_info.timestamp = timestamp;
      } else {
        break;
      }

      anno_file.getline(line, 1024, '\n');
      v_line = line + space;
      if (std::strncmp(v_line.c_str(), "sensor2lidar_rotation:", 22) == 0) {
        auto l = std::string(v_line).substr(22);
        LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "sweep s2lr:" << l;
        // std::cout << "s2lr:" << l << std::endl;
        read_s2lr_from_line(l, sweeps[cnt].cam_info);
      } else {
        break;
      }

      anno_file.getline(line, 1024, '\n');
      v_line = line + space;
      if (std::strncmp(v_line.c_str(), "sensor2lidar_translation:", 25) == 0) {
        auto l = std::string(v_line).substr(25);
        // std::cout << "s2lt:" << l << std::endl;
        LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "sweep s2lt:" << l;
        read_s2lt_from_line(l, sweeps[cnt].cam_info);
      } else {
        break;
      }
      cnt++;
    }
  }
}

static void read_cams(std::ifstream& anno_file, const std::string& path_prefix,
                      std::vector<CamInfo>& cam_infos,
                      std::vector<cv::Mat>& images, int space = 2) {
  char line[1024];
  std::string v_line;  // valid line
  // read cams
  if (anno_file.getline(line, 1024, '\n') &&
      std::strncmp(line, "cams:", 5) == 0) {
    auto s = std::string(line + 5);
    int num = std::atoi(s.c_str());
    // std::cout << "sweep num:" << num << std::endl;
    LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "cams num:" << num;
    if (num == 0) {
      return;
    }
    int cnt = 0;
    images.clear();
    cam_infos.clear();
    images.resize(num);
    cam_infos.resize(num);
    while (cnt < num) {
      anno_file.getline(line, 1024, '\n');
      v_line = line + space;
      // read data path
      if (std::strncmp(v_line.c_str(), "data_path:", 10) == 0) {
        auto data_path = std::string(v_line).substr(10);
        data_path = path_prefix + data_path;
        LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "cam data_path:" << data_path;
        if (g_use_png) {
          data_path.replace(data_path.end() - 3, data_path.end(), "png");
          LOG_IF(INFO, ENV_PARAM(DEBUG_READ))
              << "use new cam data_path:" << data_path;
        }
        // std::cout << "data_path:" << data_path << std::endl;
        images[cnt] = cv::imread(data_path);
        LOG_IF(INFO, ENV_PARAM(DEBUG_READ))
            << "cam ori size:" << images[cnt].cols << "*" << images[cnt].rows
            << "*" << images[cnt].channels();

        //if (!g_use_png && ENV_PARAM(DEBUG_BEV_RESIZE)) {
        if (!g_use_png) {
          cv::resize(images[cnt], images[cnt], cv::Size{576, 320});
          LOG_IF(INFO, ENV_PARAM(DEBUG_READ))
              << "cam size:" << images[cnt].cols << "*" << images[cnt].rows
              << "*" << images[cnt].channels();
        }
      } else {
        break;
      }

      anno_file.getline(line, 1024, '\n');
      v_line = line + space;
      if (std::strncmp(v_line.c_str(), "sensor2lidar_rotation:", 22) == 0) {
        auto l = std::string(v_line).substr(22);
        // std::cout << "s2lr:" << l << std::endl;
        LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "cam s2lr:" << l;
        read_s2lr_from_line(l, cam_infos[cnt]);
      } else {
        break;
      }

      anno_file.getline(line, 1024, '\n');
      v_line = line + space;
      if (std::strncmp(v_line.c_str(), "sensor2lidar_translation:", 25) == 0) {
        auto l = std::string(v_line).substr(25);
        // std::cout << "s2lt:" << l << std::endl;
        LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "cam s2lt:" << l;
        read_s2lt_from_line(l, cam_infos[cnt]);
      } else {
        break;
      }

      anno_file.getline(line, 1024, '\n');
      v_line = line + space;
      if (std::strncmp(v_line.c_str(), "cam_intrinsic:", 14) == 0) {
        auto l = std::string(v_line).substr(14);
        // std::cout << "s2lt:" << l << std::endl;
        LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "cam intr:" << l;
        read_cam_intr_from_line(l, cam_infos[cnt]);
      } else {
        break;
      }

      cnt++;
    }
  }
}



void read_inno_file_v2(const std::string& data_root, const std::string& seq, PointsInfoV2& points_info,
                              int points_dim, std::vector<SweepInfo>& sweeps,
                              int sweeps_points_dim,
                              std::vector<cv::Mat>& images) {
  const std::string info_dir = data_root + "/infos";
  const std::string bev_dir = data_root + "/scene_0_bev";

  auto info_path = info_dir + "/" + seq + ".info";

  std::string path_prefix = data_root + "/";
  // if (file_path.find_last_of('/') != std::string::npos) {
  //  path_prefix = file_name.substr(0, file_name.find_last_of('/') + 1);
  //  // std::cout << "path_prefix:" << path_prefix << std::endl;
  //}

  auto anno_file = std::ifstream(info_path);
  if (!anno_file) {
    std::cerr << "open:" << info_path << " fail!" << std::endl;
    exit(-1);
  }

  char line[1024];
  std::string lidar_path;
  while (anno_file.getline(line, 1024, '\n')) {
    auto len = std::strlen(line);
    if (len == 0) {
      continue;
    }

    // 1. read lidar path
    if (std::strncmp(line, "lidar_path:", 11) == 0) {
      lidar_path = std::string(line).substr(11);
      // std::cout << "lidar_path:" << lidar_path << std::endl;
      LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "lidar_path:" << lidar_path;
      // read_points_file(lidar_path, points_info);
      if (!points_info.points.points) {
        points_info.points.points.reset(new std::vector<float>);
      }
      read_points_file(path_prefix + lidar_path, *(points_info.points.points));
      points_info.points.dim = points_dim;
    } else {
      break;
    }

    // 2. read timestamp
    if (anno_file.getline(line, 1024, '\n') &&
        std::strncmp(line, "timestamp:", 10) == 0) {
      auto timestamp = std::atoll(std::string(line).substr(10).c_str());
      // std::cout << "timestamp:" << timestamp<< std::endl;
      LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "timestamp:" << timestamp;
      // points_info.cam_info.timestamp = timestamp;
      points_info.timestamp = timestamp;
    } else {
      break;
    }

    // 3. read sweeps
    read_sweeps(anno_file, path_prefix, sweeps, sweeps_points_dim);

    // 4. read cams
    read_cams(anno_file, path_prefix, points_info.cam_info, images);
    break;
  }
  anno_file.close();

  std::string lidar_base_name;
  if (lidar_path.find_last_of('/') != std::string::npos) {
    lidar_base_name = lidar_path.substr(lidar_path.find_last_of('/') + 1);
  }
  auto bev_path = bev_dir + "/" +
                  lidar_base_name.substr(0, lidar_base_name.find('.')) + ".jpg";
  LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "lidar basename:" << lidar_base_name;
  LOG_IF(INFO, ENV_PARAM(DEBUG_READ)) << "bev path:" << bev_path;

  // 5. read bev
  points_info.bev = cv::imread(bev_path);
  LOG_IF(INFO, ENV_PARAM(DEBUG_READ))
      << "bev ori size:" << points_info.bev.cols << "*" << points_info.bev.rows
      << "*" << points_info.bev.channels();
  //if (ENV_PARAM(DEBUG_BEV_RESIZE)) {

  cv::resize(points_info.bev, points_info.bev, cv::Size{400, 400});
  //}
  if (points_info.bev.empty()) {
    LOG(ERROR) << "read bev:" << bev_path << " error!";
    abort();
  }
}

void print_points_info(const PointsInfo& points_info) {
  std::cout << "points info: " << std::endl;
  std::cout << "  cam_info:" << std::endl;
  for (auto n = 0u; n < points_info.cam_info.size(); ++n) {
    std::cout << "    timestamp:" << points_info.cam_info[n].timestamp
              << std::endl;
    std::cout << "    s2l_t:";
    for (auto i = 0u; i < points_info.cam_info[n].s2l_t.size(); ++i) {
      std::cout << points_info.cam_info[n].s2l_t[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "    s2l_r:";

    for (auto i = 0u; i < points_info.cam_info[n].s2l_r.size(); ++i) {
      std::cout << points_info.cam_info[n].s2l_r[i] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "  dim:" << points_info.points.dim << std::endl;
  if (points_info.points.points) {
    std::cout << "  points size:" << points_info.points.points->size()
              << std::endl;
  }
}

}}}
