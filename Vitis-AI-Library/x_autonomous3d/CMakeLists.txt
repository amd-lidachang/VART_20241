#
# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
vai_add_library(
  SRCS ${LIST_PBSRC_CC}
       src/x_autonomous3d.cpp
       src/x_autonomous3d_imp.hpp
       src/x_autonomous3d_imp.cpp
       src/voxelizer.hpp
       src/voxelizer.cpp
       src/voxelizer_cpt_waymo.hpp
       src/voxelizer_cpt_waymo.cpp
       src/scatter.hpp
       src/scatter.cpp
       #src/waymo_preprocess.cpp
       #src/waymo_preprocess.hpp
  PUBLIC_REQUIRE ${PROJECT_NAME}::xnnpp ${PROJECT_NAME}::dpu_task xir::xir)

vai_add_test(test_x_autonomous3d)
vai_add_test(test_x_autonomous3d_batch)
vai_add_test(test_x_autonomous3d_scatter)
#vai_add_test(test_bin_x_autonomous3d)
vai_add_test(test_accuracy_x_autonomous3d)

