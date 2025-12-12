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
#include <vector>

using std::vector;

namespace vitis {
namespace ai {
namespace centerpoint_waymo {

int atomicAdd_logic(int *address, int val);
//void generate_pillars_gpu(at::Tensor points,
//    const int in_num_points, const int max_num_pillars, const int max_num_points_per_pillar, const int num_point_feature,
//    const int grid_x_size, const int grid_y_size, const int grid_z_size,
//    const float min_x_range, const float min_y_range, const float min_z_range,
//    const float pillar_x_size, const float pillar_y_size, const float pillar_z_size,
//    at::Tensor pillar_point_feature_in_coors,
//    at::Tensor pillar_count_histo, at::Tensor counter, at::Tensor pillar_count,
//    at::Tensor x_coors, at::Tensor y_coors,
//    at::Tensor num_points_per_pillar, at::Tensor pillar_point_feature, at::Tensor pillar_coors,
//    at::Tensor pillar_point_mean,
//    at::Tensor pillar_pfn_feature);
//
void generate_pillars(const vector<float> &points, // [n, 6]
    // in_num_points : n,  max_num_pillars: 60000, max_num_points_per_pillar: 20, num_point_feature: 6
    const int in_num_points, const int max_num_pillars, const int max_num_points_per_pillar, const int num_point_feature,
    const int grid_x_size, const int grid_y_size, const int grid_z_size,
    // [-74.88, -74.88, -2]
    const float min_x_range, const float min_y_range, const float min_z_range,
    // voxel_size [0.24, 0.24, 6]
    const float pillar_x_size, const float pillar_y_size, const float pillar_z_size,
    // [624, 624, 20, 6]
    vector<float> &pillar_point_feature_in_coors,
    // [624, 624]
    vector<int> &pillar_count_histo, 
    int &counter, int &pillar_count, // output num
    // 60000, 60000
    vector<int> &x_coors, vector<int> &y_coors,
    // 60000 : output, [60000, 20, 6], [60000, 4]: output 
    vector<int> &num_points_per_pillar, vector<float> &pillar_point_feature, vector<int> &pillar_coors,
    // [60000, 1, 3]
    vector<float> &pillar_point_mean,
    // [60000, 20, 11] : output
    vector<float> &pillar_pfn_feature);

void voxel_reorder(vector<float> &feature, int pillar_count, vector<int> &coors, vector<int> &num_per_pillar, const vector<int> &normal_coors);

void set_input(const vector<float> &feature, float scale, int8_t *input_tensor, int tensor_size);

void normalize_set_input(const vector<float> &feature, const vector<float> scales, float tensor_scale, int8_t *input_tensor, int tensor_size);

void normalize(const vector<float> &feature, const vector<float> scales, vector<float> &output);

void save_vector(const std::string& name, const float *input, int dim, int line);
void save_vector(const std::string& name, const int *input, int dim, int line);
void write_bin(const std::string& name, const int *input, int size); 
void write_bin(const std::string& name, const float *input, int size); 

}}}
