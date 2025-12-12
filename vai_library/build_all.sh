#!/bin/bash
#
# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -e

script_path=$(dirname "$(realpath $0)")
project_name=$(basename ${script_path})

cd -P "$script_path";

(
    cd ..;
    [ -d unilog ] ||  git clone gits@xcdl190260:aisw/unilog
    [ -d xir ] ||  git clone gits@xcdl190260:aisw/xir
    [ -d target_factory ] ||  git clone gits@xcdl190260:aisw/target_factory
    [ -d vart ] ||  git clone gits@xcdl190260:aisw/vart
)

(cd ../unilog; "$script_path"/cmake.sh --project unilog "$@")
(cd ../target_factory; "$script_path"/cmake.sh --project target_factory "$@")
(cd ../xir; "$script_path"/cmake.sh --project xir --build-python "$@")
(cd ../vart; "$script_path"/cmake.sh --project vart --build-python --cmake-options='-DENABLE_DPU_RUNNER=ON' --cmake-options='-DENABLE_SIM_RUNNER=ON' --cmake-options='-DENABLE_CPU_RUNNER=ON'   "$@")
(cd .; "$script_path"/cmake.sh --project Vitis-AI-Library --build-python --cmake-options="-DENABLE_OVERVIEW=ON" "$@")
