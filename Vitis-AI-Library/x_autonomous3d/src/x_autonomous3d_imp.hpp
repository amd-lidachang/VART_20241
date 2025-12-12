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

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/x_autonomous3d.hpp>
#include "./voxelizer.hpp"

namespace vitis {
namespace ai {

class X_Autonomous3DImp : public X_Autonomous3D {
 public:
  X_Autonomous3DImp(const std::string& model_name_0,
                    const std::string& model_name_1);
  virtual ~X_Autonomous3DImp();

  virtual X_Autonomous3DResult run(const std::vector<float>& input) override;
  virtual std::vector<X_Autonomous3DResult> run(
      const std::vector<std::vector<float>>& inputs) override;

  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;
  virtual size_t get_input_batch() const override;

 private:
  X_Autonomous3DResult run_simple(const std::vector<float>& input);
  std::vector<X_Autonomous3DResult> run_internal(
      const std::vector<std::vector<float>>& inputs);

  void reset(int8_t* model_0_ptr, int model_0_size, int8_t* model_1_ptr,
             int model_1_size);
  // uint32_t points_dim_;
  std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_0_;
  std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_1_;
  std::unique_ptr<vitis::ai::X_Autonomous3DPostProcess> postprocessor_;

  int max_num_pillars;            //  = 60000;
  int max_num_points_per_pillar;  // = 20;
  int num_point_feature;          //  = 6;
  int grid_x_size;                // = 624;
  int grid_y_size;                // = 624;
  int grid_z_size;                // = 1;
  float min_x_range;              // = -74.88;
  float min_y_range;              // = -74.88;
  float min_z_range;              // = -2;
  float pillar_x_size;            // = 0.24;
  float pillar_y_size;            // = 0.24;
  float pillar_z_size;            // = 6.0;
  // vector<float> pillar_point_feature_in_coors; //(624 * 624 * 20 * 6);
  // vector<int> pillar_count_histo; //(624 * 624);
  int counter;       // = 0;
  int pillar_count;  //= 0;
  // vector<int> x_coors; //(60000, 0.f);
  // vector<int> y_coors; //(60000, 0.f);
  // vector<int> num_points_per_pillar; //(60000, 0.f);
  // vector<float> pillar_point_feature; // (60000 * 20 * 6, 0.f);
  // vector<int> pillar_coors; //(60000 * 4, 0.f);
  // vector<float> pillar_point_mean; //(60000 * 1 * 3, 0.f);
  // vector<float> pillar_pfn_feature; //(60000 * 20 * 11, 0.f);
  std::unique_ptr<x_autonomous3d::Voxelizer> voxelizer_;
  x_autonomous3d::VoxelConfig voxel_config_;
};

}  // namespace ai
}  // namespace vitis

