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
#include <memory>
//#include <thread>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "./voxelizer.hpp"
#include "./voxelizer_cpt_waymo.hpp"

namespace vitis {
namespace ai {
namespace x_autonomous3d {

DEF_ENV_PARAM(DEBUG_VOXELIZE, "0");

std::unique_ptr<Voxelizer> Voxelizer::create(const VoxelConfig& config) {
  switch (config.type) {
    case VoxelType::CENTERPOINT_WAYMO:
      return std::unique_ptr<Voxelizer>(new VoxelizerCptWaymo(config));
      // return nullptr;
      break;
    default:
      return nullptr;
  }
  return nullptr;
}

void print_config(const VoxelConfig& config) {
  std::cout << "print voxel config:" << std::endl;
  std::cout << "input means:";
  for (auto i = 0u; i < config.input_means.size(); ++i) {
    std::cout << config.input_means[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "input scales:";
  for (auto i = 0u; i < config.input_scales.size(); ++i) {
    std::cout << config.input_scales[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "max points num:" << config.max_points_num << std::endl;
  std::cout << "max voxels num:" << config.max_voxels_num << std::endl;
  std::cout << "feature dim:" << config.feature_dim << std::endl;
  std::cout << "coors dim:" << config.coors_dim << std::endl;
  std::cout << "in channels:" << config.in_channels << std::endl;

  std::cout << "voxels size:";
  for (auto i = 0u; i < config.voxels_size.size(); ++i) {
    std::cout << config.voxels_size[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "coors range:";
  for (auto i = 0u; i < config.coors_range.size(); ++i) {
    std::cout << config.coors_range[i] << " ";
  }
  std::cout << std::endl;
}

Voxelizer::Voxelizer(const VoxelConfig& config) : config_(config) {
  if (ENV_PARAM(DEBUG_VOXELIZE)) {
    print_config(config);
  }
}

Voxelizer::~Voxelizer() {}

}  // namespace x_autonomous3d
}  // namespace ai
}  // namespace vitis

