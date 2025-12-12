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
#include <cstring>
#include <iostream>
#include <memory>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "./scatter.hpp"
#include "./waymo_preprocess.hpp"

#include "./x_autonomous3d_imp.hpp"

using namespace std;
namespace vitis {
namespace ai {

DEF_ENV_PARAM(DEBUG_REORDER, "0");
DEF_ENV_PARAM(DEBUG_LOAD_PRE_MIDDLE_INPUT, "0");
DEF_ENV_PARAM(DEBUG_SAVE, "0");
DEF_ENV_PARAM(DEBUG_LOAD_MIDDLE_INPUT, "0");
DEF_ENV_PARAM(DEBUG_LOAD_MIDDLE_OUTPUT, "0");
DEF_ENV_PARAM(SAVE_PREPROCESS, "0");
DEF_ENV_PARAM(DEBUG_SIMPLE_PREPROCESS, "0");
DEF_ENV_PARAM(DEBUG_X_AUTONOMOUS3D, "0");

X_Autonomous3DImp::X_Autonomous3DImp(const std::string& model_name_0,
                                     const std::string& model_name_1)
    :  // points_dim_(4),
      model_0_{ConfigurableDpuTask::create(model_name_0, false)},
      model_1_{ConfigurableDpuTask::create(model_name_1, false)},

      postprocessor_{vitis::ai::X_Autonomous3DPostProcess::create(
          model_1_->getInputTensor()[0], model_1_->getOutputTensor()[0],
          model_1_->getConfig())} {
  max_num_pillars = 60000;
  max_num_points_per_pillar = 20;
  num_point_feature = 6;
  grid_x_size = 624;
  grid_y_size = 624;
  grid_z_size = 1;
  min_x_range = -74.88;
  min_y_range = -74.88;
  min_z_range = -2;
  pillar_x_size = 0.24;
  pillar_y_size = 0.24;
  pillar_z_size = 6.0;
  // pillar_point_feature_in_coors.resize(624 * 624 * 20 * 6);
  // pillar_count_histo.resize(624 * 624);
  // counter = 0;
  // pillar_count = 0;
  // x_coors.resize(60000, 0.f);
  // y_coors.resize(60000, 0.f);
  // num_points_per_pillar.resize(60000);
  // pillar_point_feature.resize(60000 * 20 * 6);
  // pillar_coors.resize(60000 * 4);
  // pillar_point_mean.resize(60000 * 1 * 3);
  // pillar_pfn_feature.resize(60000 * 20 * 11);
  voxel_config_.type = x_autonomous3d::VoxelType::CENTERPOINT_WAYMO;
  vector<float> scales{74.88, 74.88, 4.0, 1, 1.5, 1, 1, 1, 6.0, 1, 1};
  voxel_config_.input_scales = scales;
  voxel_config_.max_points_num = max_num_points_per_pillar;
  voxel_config_.max_voxels_num = max_num_pillars;
  voxel_config_.feature_dim = 11;
  voxel_config_.voxels_size = {pillar_x_size, pillar_y_size, pillar_z_size};
  voxel_config_.coors_range = {-74.88, -74.88, -2, 74.88, 74.88, 4};
  voxel_config_.coors_dim = 4;
  voxel_config_.in_channels = 64;
  voxelizer_ = x_autonomous3d::Voxelizer::create(voxel_config_);
}

X_Autonomous3DImp::~X_Autonomous3DImp() {}

int X_Autonomous3DImp::getInputWidth() const {
  return model_0_->getInputWidth();
}

int X_Autonomous3DImp::getInputHeight() const {
  return model_0_->getInputHeight();
}

size_t X_Autonomous3DImp::get_input_batch() const {
  return model_0_->get_input_batch();
}

X_Autonomous3DResult X_Autonomous3DImp::run_simple(
    const std::vector<float>& input) {
  __TIC__(CENTERPOINT_E2E)
  auto batch_size = get_input_batch();
  // 1. preprocess
  __TIC__(CENTERPOINT_PREPROCESS)

  auto model_0_input_size_per_batch =
      model_0_->getInputTensor()[0][0].size / batch_size;
  auto model_0_input_ptr =
      (int8_t*)model_0_->getInputTensor()[0][0].get_data(0);
  auto input_tensor_scale =
      vitis::ai::library::tensor_scale(model_0_->getInputTensor()[0][0]);
  LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D))
      << "model_0 input tensor scale:" << input_tensor_scale;
  LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D))
      << "model_0 input pointer:" << (void*)model_0_input_ptr;
  auto pillar_coors =
      voxelizer_->voxelize(input, 6, model_0_input_ptr, input_tensor_scale,
                           model_0_input_size_per_batch);
  int voxels_num = pillar_coors.size() / 4;
  LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D)) << "voxels num:" << voxels_num;
  __TOC__(CENTERPOINT_PREPROCESS)

  //  2. run model 0
  __TIC__(CENTERPOINT_DPU_0)
  model_0_->run(0);
  __TOC__(CENTERPOINT_DPU_0)

  // 3. middle process
  __TIC__(CENTERPOINT_MIDDLE_PROCESS)
  auto in_channels = 64;
  auto nx = 624;
  auto ny = 624;
  // vitis::ai::centerpoint::middle_process(pillar_coors,
  LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D))
      << "model 0 output tensor scale:"
      << vitis::ai::library::tensor_scale(model_0_->getOutputTensor()[0][0]);
  LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D))
      << "model 1 input tensor scale:"
      << vitis::ai::library::tensor_scale(model_1_->getInputTensor()[0][0]);

  vitis::ai::x_autonomous3d::scatter(
      pillar_coors, 4, (int8_t*)model_0_->getOutputTensor()[0][0].get_data(0),
      vitis::ai::library::tensor_scale(model_0_->getOutputTensor()[0][0]),
      (int8_t*)model_1_->getInputTensor()[0][0].get_data(0),
      vitis::ai::library::tensor_scale(model_1_->getInputTensor()[0][0]),
      in_channels, nx, ny);

  __TOC__(CENTERPOINT_MIDDLE_PROCESS)

  // 4. run model 1
  __TIC__(CENTERPOINT_DPU_1)
  model_1_->run(0);
  __TOC__(CENTERPOINT_DPU_1)

  // 5. post process
  __TIC__(CENTERPOINT_POSTPROCESS)
  auto bbox_finals = postprocessor_->process(1u);
  __TOC__(CENTERPOINT_POSTPROCESS)
  __TOC__(CENTERPOINT_E2E)
  return bbox_finals[0];
}

X_Autonomous3DResult X_Autonomous3DImp::run(const std::vector<float>& input) {
  // return run_simple(input);
  vector<vector<float>> batch_input(1, input);
  auto result = run_internal(batch_input);
  return result[0];
}

std::vector<X_Autonomous3DResult> X_Autonomous3DImp::run_internal(
    const std::vector<std::vector<float>>& batch_inputs) {
  __TIC__(X_AUTONOMOUS3D_E2E)
  size_t batch = get_input_batch();
  auto num = std::min(batch, batch_inputs.size());
  LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D)) << "batch:" << batch;
  LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D)) << "num:" << num;
  auto model_0_input_size = model_0_->getInputTensor()[0][0].size / batch;

  LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D))
      << "model_0 input size:" << model_0_input_size;

  auto model_0_input_scale =
      vitis::ai::library::tensor_scale(model_0_->getInputTensor()[0][0]);

  LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D))
      << "model_0 input scale:" << model_0_input_scale;
  // todo: multi threads
  __TIC__(X_AUTONOMOUS3D_PREPROCESS)
  std::vector<vector<int>> batch_coors(num);
  for (auto i = 0u; i < num; ++i) {
    std::memset(model_0_->getInputTensor()[0][0].get_data(i), 0,
                model_0_input_size);
  }
  for (auto i = 0u; i < num; ++i) {
    auto input_ptr = (int8_t*)model_0_->getInputTensor()[0][0].get_data(i);
    LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D))
        << "model_0 input pointer:" << (void*)input_ptr;
    batch_coors[i] = voxelizer_->voxelize(
        batch_inputs[i], 6, input_ptr, model_0_input_scale, model_0_input_size);
  }
  __TOC__(X_AUTONOMOUS3D_PREPROCESS)

  __TIC__(X_AUTONOMOUS3D_DPU_0)
  model_0_->run(0);
  __TOC__(X_AUTONOMOUS3D_DPU_0)

  __TIC__(X_AUTONOMOUS3D_MIDDLE_PROCESS)
  auto model_1_input_tensor_size = model_1_->getInputTensor()[0][0].size;
  auto model_1_input_size = model_1_input_tensor_size / batch;
  LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D))
      << "model_1 input tensor size:" << model_1_input_tensor_size
      << " model_1 input size:" << model_1_input_size;
  for (auto i = 0u; i < num; ++i) {
    std::memset(model_1_->getInputTensor()[0][0].get_data(i), 0,
                model_1_input_size);
  }

  auto coors_dim = voxel_config_.coors_dim;
  auto nx = model_1_->getInputTensor()[0][0].width;
  auto ny = model_1_->getInputTensor()[0][0].height;
  auto in_channels = model_1_->getInputTensor()[0][0].channel;
  LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D))
      << "nx: " << nx << ", ny: " << ny << ", in_channels:" << in_channels;
  for (auto i = 0u; i < num; ++i) {
    vitis::ai::x_autonomous3d::scatter(
        batch_coors[i], coors_dim,
        (int8_t*)model_0_->getOutputTensor()[0][0].get_data(i),
        vitis::ai::library::tensor_scale(model_0_->getOutputTensor()[0][0]),
        (int8_t*)model_1_->getInputTensor()[0][0].get_data(i),
        vitis::ai::library::tensor_scale(model_1_->getInputTensor()[0][0]),
        in_channels, nx, ny);
  }

  __TOC__(X_AUTONOMOUS3D_MIDDLE_PROCESS)

  __TIC__(X_AUTONOMOUS3D_DPU_1)
  model_1_->run(0);
  __TOC__(X_AUTONOMOUS3D_DPU_1)

  // todo : multi threads
  __TIC__(CLOCS_POINTPILLARS_POSTPROCESS)
  auto bbox_finals = postprocessor_->process(num);
  __TOC__(CLOCS_POINTPILLARS_POSTPROCESS)

  __TOC__(X_AUTONOMOUS3D_E2E)
  return bbox_finals;
}

std::vector<X_Autonomous3DResult> X_Autonomous3DImp::run(
    const std::vector<std::vector<float>>& inputs) {
  return run_internal(inputs);
}

}  // namespace ai
}  // namespace vitis

