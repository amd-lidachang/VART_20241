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

#include "./x_autonomous3d_imp.hpp"
#include "vitis/ai/x_autonomous3d.hpp"

namespace vitis {
namespace ai {

X_Autonomous3D::X_Autonomous3D() {}
X_Autonomous3D::~X_Autonomous3D() {}

std::unique_ptr<X_Autonomous3D> X_Autonomous3D::create(
    const std::string& model_name_0, const std::string& model_name_1) {
  return std::unique_ptr<X_Autonomous3D>(
      new X_Autonomous3DImp(model_name_0, model_name_1));
}

}  // namespace ai
}  // namespace vitis

