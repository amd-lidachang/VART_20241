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
#include "./voxelizer.hpp"

using namespace std;

namespace vitis {
namespace ai {
namespace x_autonomous3d {

class VoxelizerCptWaymo : public Voxelizer {
 public:
  virtual ~VoxelizerCptWaymo();

  virtual std::vector<int> voxelize(const vector<float>& points, int dim,
                                    int8_t* input_tensor_ptr,
                                    float tensor_scale,
                                    size_t input_tensor_size) override;

  explicit VoxelizerCptWaymo(const VoxelConfig& config);

 private:
  int voxelize_normal_core_simple(const vector<float>& points, int dim,
                                  vector<float>& pillar_point_feature);

  std::vector<int> voxelize_input_simple(const vector<float>& points, int dim,
                                         float tensor_scale,
                                         int8_t* input_tensor_ptr,
                                         size_t input_tensor_size);
  int voxelize_padding_simple(const vector<float>& points, int pillar_num,
                              int dim, int8_t* input_tensor_ptr,
                              int input_tensor_dim,
                              const vector<float>& input_scales,
                              vector<int>& coors, int coors_dim);

 private:
  // [624, 624, 20, 6]
  vector<float> pillar_point_feature_in_coors_;
  // [624, 624]
  vector<int> pillar_count_histo_;
  int counter_;
  int pillar_count_;  // output num

  // 60000, 60000
  vector<int> x_coors_;
  vector<int> y_coors_;
  // 60000 : output, [60000, 20, 6], [60000, 4]: output
  vector<int> num_points_per_pillar_;
  vector<float> pillar_point_feature_;
  // [60000, 1, 3]
  vector<float> pillar_point_mean_;
  // [60000, 20, 11] : output
  vector<float> pillar_pfn_feature_;

  // vector<int> coors_;

  // [60000, 20]
  vector<int> pillar_point_ori_indices_;
  // [624, 624]
  vector<int> grid_fake_index_map_;  // value : index of 60000
  // [624, 624]
  vector<int>
      grid_points_count_map_;  // key : y * 624 + x ; value : index of 60000
};

}  // namespace x_autonomous3d
}  // namespace ai
}  // namespace vitis

