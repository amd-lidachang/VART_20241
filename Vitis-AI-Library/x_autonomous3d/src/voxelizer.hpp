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
#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace vitis {
namespace ai {
namespace x_autonomous3d {

// constexpr uint32_t MAX_POINTS_NUM = 64;
// constexpr uint32_t MAX_VOXELS_NUM = 40000;
enum VoxelType {
  POINTPILLARS = 0,
  POINTPILLARS_NUS = 1,
  CENTERPOINT_WAYMO = 2,
};

struct VoxelConfig {
  VoxelType type;
  std::vector<float> input_means;
  std::vector<float> input_scales;
  int max_points_num;              // 20
  int max_voxels_num;              // 60000
  int feature_dim;                 // 11
  std::vector<float> voxels_size;  // [0.24, 0.24, 6]
  std::vector<float> coors_range;  // [-74.88, -74.88, -2, 74.88, 74.88, 4]
  int coors_dim;                   // 4
  int in_channels;                 // 64
};

class Voxelizer {
 public:
  static std::unique_ptr<Voxelizer> create(const VoxelConfig& config);

  virtual ~Voxelizer();

  virtual std::vector<int> voxelize(const vector<float>& points, int dim,
                                    int8_t* input_tensor_ptr,
                                    float tensor_scale,
                                    size_t input_tensor_size) = 0;

 protected:
  explicit Voxelizer(const VoxelConfig& config);
  VoxelConfig config_;
};

}  // namespace x_autonomous3d
}  // namespace ai
}  // namespace vitis

