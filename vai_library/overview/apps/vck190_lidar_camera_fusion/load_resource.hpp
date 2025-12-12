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

#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <vitis/ai/pointpainting.hpp>
#include <vector>

using vitis::ai::pointpillars_nus::SweepInfo;

namespace vitis {
namespace ai {
namespace pointpainting {

struct PointsInfoV2 : public vitis::ai::pointpillars_nus::PointsInfo {
  cv::Mat bev;
  std::vector<cv::Mat> cams;
};

struct PtFrameInfo {
  int channel_id;
  unsigned long frame_id;
  PointsInfoV2 pt_info;
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
};


void load(const std::string& data_root, const std::vector<std::string>& seq_list,
          std::vector<PointsInfoV2>& points_infos, int start, int end);

void read_inno_file_v2(const std::string& data_root, const std::string& seq, PointsInfoV2& points_info,
                              int points_dim, std::vector<SweepInfo>& sweeps,
                              int sweeps_points_dim,
                              std::vector<cv::Mat>& images);




}}}




