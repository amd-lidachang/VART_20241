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
#include "./voxelizer_cpt_waymo.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
DEF_ENV_PARAM(DEBUG_CENTERPOINT_WAYMO, "0");
DEF_ENV_PARAM(DEBUG_VOXEL, "0");

namespace vitis {
namespace ai {
namespace x_autonomous3d {

VoxelizerCptWaymo::VoxelizerCptWaymo(const VoxelConfig& config)
    : Voxelizer(config), counter_(0), pillar_count_(0) {
  // pillar_point_feature_in_coors_.resize(624 * 624 * 20 * 6);
  // pillar_count_histo_.resize(624 * 624);
  // x_coors_.resize(60000);
  // y_coors_.resize(60000);
  num_points_per_pillar_.resize(60000);
  // pillar_point_feature_.resize(60000 * 20 * 6);
  // pillar_point_mean_.resize(60000 * 1 * 3);
  // pillar_pfn_feature_.resize(60000 * 20 * 11);
  // coors_.resize(4 * 60000);

  pillar_point_ori_indices_.resize(60000 * 20);
  grid_fake_index_map_.resize(624 * 624);
  grid_points_count_map_.resize(
      624 * 624);  // key : y * 624 + x ; value : index of 60000
}

VoxelizerCptWaymo::~VoxelizerCptWaymo() {}

int VoxelizerCptWaymo::voxelize_normal_core_simple(
    const vector<float>& points, int dim, vector<float>& pillar_point_feature) {
  int grid_x_size = 624;
  int grid_y_size = 624;
  int grid_z_size = 1;
  float min_x_range = -74.88;
  float min_y_range = -74.88;
  float min_z_range = -2;
  float pillar_x_size = 0.24;
  float pillar_y_size = 0.24;
  float pillar_z_size = 6;

  // reset pillar_count_
  pillar_count_ = 0;

  auto max_pillar_num = config_.max_voxels_num;
  auto max_point_num = config_.max_points_num;
  __TIC__(RECORD_INDEX)
  // 1. record indexes
  // num_points_per_pillar_.assign(num_points_per_pillar_.size(), 0);
  int all_points_num = points.size() / dim;
  pillar_point_ori_indices_.assign(pillar_point_ori_indices_.size(), 0);
  grid_fake_index_map_.assign(grid_fake_index_map_.size(),
                              -1);  // value : index of 60000
  grid_points_count_map_.assign(
      grid_points_count_map_.size(),
      0);  // key : y * 624 + x ; value : index of 60000

  for (auto point_i = 0; point_i < all_points_num; ++point_i) {
    int pos = point_i * dim;
    int y_coor = std::floor((points[pos + 1] - min_y_range) / pillar_y_size);
    int x_coor = std::floor((points[pos + 0] - min_x_range) / pillar_x_size);
    int z_coor = std::floor((points[pos + 2] - min_z_range) / pillar_z_size);
    if (x_coor >= 0 && x_coor < grid_x_size && y_coor >= 0 &&
        y_coor < grid_y_size && z_coor >= 0 && z_coor < grid_z_size) {
      // auto grid = std::make_pair(y_coor, x_coor);
      auto grid = y_coor * 624 + x_coor;
      if (grid_points_count_map_[grid] >= max_point_num) {
        continue;
      }
      // if (!grid_index_map.count(grid)) {
      if (grid_points_count_map_[grid] == 0 && pillar_count_ < max_pillar_num) {
        grid_fake_index_map_[grid] = pillar_count_;
        pillar_count_++;
      } else if (pillar_count_ >= max_pillar_num) {
        continue;
      }
      pillar_point_ori_indices_[grid_fake_index_map_[grid] * max_point_num +
                                grid_points_count_map_[grid]] = point_i;
      grid_points_count_map_[grid]++;
    }
  }
  __TOC__(RECORD_INDEX)
  return pillar_count_;
}

int VoxelizerCptWaymo::voxelize_padding_simple(
    const vector<float>& points, int pillar_num, int dim,
    int8_t* input_tensor_ptr, int input_tensor_dim,
    const vector<float>& input_scales, vector<int>& coors, int coors_dim) {
  // vector<float> pillar_point_mean(pillar_num * 3);
  vector<float> pillar_point_mean(3);
  int max_point_num = config_.max_points_num;
  num_points_per_pillar_.assign(num_points_per_pillar_.size(), 0);

  if (coors.size() != (uint32_t)(60000 * coors_dim)) {
    coors.resize(60000 * coors_dim);
  }
  coors.assign(coors.size(), 0);

  float min_x_range = -74.88;
  float min_y_range = -74.88;
  // float min_z_range = -2;
  float pillar_x_size = 0.24;
  float pillar_y_size = 0.24;
  // float pillar_z_size = 6;
  float x_offset = pillar_x_size / 2 + min_x_range;
  float y_offset = pillar_y_size / 2 + min_y_range;

  int pillar_count = 0;
  for (auto i = 0u; i < grid_fake_index_map_.size(); ++i) {
    auto point_num = grid_points_count_map_[i];
    if (point_num == 0) continue;

    num_points_per_pillar_[pillar_count] = point_num;
    auto pillar_fake_index = grid_fake_index_map_[i];

    // mean calculate
    pillar_point_mean.assign(3, 0);
    for (auto point_i = 0; point_i < point_num; ++point_i) {
      int src_pos =
          pillar_point_ori_indices_[pillar_fake_index * max_point_num +
                                    point_i];
      pillar_point_mean[0] += points[src_pos * dim + 0];
      pillar_point_mean[1] += points[src_pos * dim + 1];
      pillar_point_mean[2] += points[src_pos * dim + 2];
    }

    for (int mean_i = 0; mean_i < 3; ++mean_i) {
      pillar_point_mean[mean_i] /= point_num;
    }

    for (int point_i = 0; point_i < point_num; ++point_i) {
      auto x_coor = i % 624;
      auto y_coor = i / 624;
      coors[pillar_count * 4 + 2] = y_coor;
      coors[pillar_count * 4 + 3] = x_coor;
      int dst_pos = pillar_count * max_point_num + point_i;
      int src_pos =
          pillar_point_ori_indices_[pillar_fake_index * max_point_num +
                                    point_i];

      // gather[:, :, 0:6] = voxel_features[:, :, 0:6]
      for (int i = 0; i < dim; ++i) {
        input_tensor_ptr[dst_pos * input_tensor_dim + i] =
            (int)((points[src_pos * dim + i]) * input_scales[i]);
      }
      // gather[:, :, 6:9] = voxel_features[:, :, :3] - points_mean
      for (auto i = 0; i < 3; ++i) {
        input_tensor_ptr[dst_pos * input_tensor_dim + dim + i] =
            (int)((points[src_pos * dim + i] - pillar_point_mean[i]) *
                  input_scales[dim + i]);
      }

      // gather[:, :, 9] = voxel_features[:, :, 0] - (coords[:,
      // 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x +
      // self.x_offset)
      input_tensor_ptr[dst_pos * input_tensor_dim + dim + 3] =
          (int)((points[src_pos * dim + 0] -
                 (coors[pillar_count * 4 + 3] * pillar_x_size + x_offset)) *
                input_scales[dim + 3]);
      // gather[:, :, 10] = voxel_features[:, :, 0] - (coords[:,
      // 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y +
      // self.y_offset)
      input_tensor_ptr[dst_pos * input_tensor_dim + dim + 4] =
          (int)((points[src_pos * dim + 1] -
                 (coors[pillar_count * 4 + 2] * pillar_y_size + y_offset)) *
                input_scales[dim + 4]);
    }

    pillar_count++;
  }
  return pillar_count;
}

std::vector<int> VoxelizerCptWaymo::voxelize_input_simple(
    const vector<float>& points, int dim, float tensor_scale,
    int8_t* input_tensor_ptr, size_t input_tensor_size) {
  __TIC__(VOXELIZE_INPUT_SIMPLE)
  __TIC__(VOXELIZE_NORMAL_CORE)
  int pillar_count =
      voxelize_normal_core_simple(points, dim, pillar_point_feature_);
  __TOC__(VOXELIZE_NORMAL_CORE)
  LOG_IF(INFO, ENV_PARAM(DEBUG_CENTERPOINT_WAYMO))
      << "pillar count:" << pillar_count;

  vector<float> valid_scales(config_.input_scales.size());
  for (auto i = 0u; i < valid_scales.size(); ++i) {
    valid_scales[i] = 1.f / config_.input_scales[i] * tensor_scale;
  }
  __TIC__(VOXELIZE_PADDING)
  int coors_dim = 4;
  std::vector<int> coors(pillar_count * coors_dim, 0);
  voxelize_padding_simple(points, pillar_count, dim, input_tensor_ptr, 11,
                          valid_scales, coors, coors_dim);
  __TOC__(VOXELIZE_PADDING)
  // save_vector("num_points_per_pillar_debug2.txt",
  // num_points_per_pillar_.data(), 1, 60000);
  __TOC__(VOXELIZE_INPUT_SIMPLE)
  // return pillar_count;
  return coors;
}

std::vector<int> VoxelizerCptWaymo::voxelize(const vector<float>& points,
                                             int dim, int8_t* input_tensor_ptr,
                                             float tensor_scale,
                                             size_t input_tensor_size) {
  // 0. clear buffer;
  num_points_per_pillar_.assign(num_points_per_pillar_.size(), 0);
  pillar_point_ori_indices_.assign(pillar_point_ori_indices_.size(), 0);
  grid_fake_index_map_.assign(grid_fake_index_map_.size(), 0);
  grid_points_count_map_.assign(grid_points_count_map_.size(), 0);

  auto coors = voxelize_input_simple(points, dim, tensor_scale,
                                     input_tensor_ptr, input_tensor_size);
  return coors;
}

}  // namespace x_autonomous3d
}  // namespace ai
}  // namespace vitis
