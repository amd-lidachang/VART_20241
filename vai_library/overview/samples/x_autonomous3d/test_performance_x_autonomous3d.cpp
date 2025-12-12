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
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vitis/ai/benchmark.hpp>
#include <vitis/ai/x_autonomous3d.hpp>

static void read_points_file(const std::string& points_file_name,
                             std::vector<float>& points) {
  struct stat file_stat;
  if (stat(points_file_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << points_file_name << " state error!" << std::endl;
    exit(-1);
  }
  auto file_size = file_stat.st_size;
  points.resize(file_size / 4);
  CHECK(std::ifstream(points_file_name)
            .read(reinterpret_cast<char*>(points.data()), file_size)
            .good());
}

namespace vitis {
namespace ai {

class X_Autonomous3DPerf {
 public:
  X_Autonomous3DPerf(std::string input_n, std::string model_1,
                     std::string model_2)
      : input_name(input_n),
        kernel_name_1(model_1),
        kernel_name_2(model_2),
        det(X_Autonomous3D::create(kernel_name_1, kernel_name_2)) {
    batch_size = get_input_batch();
    all_arrays.resize(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
      std::vector<float> array;
      read_points_file(input_name, array);
      all_arrays[i] = array;
    }
  }
  int getInputWidth() { return det->getInputWidth(); }
  int getInputHeight() { return det->getInputHeight(); }
  size_t get_input_batch() { return (size_t)det->get_input_batch(); }
  std::vector<X_Autonomous3DResult> run(const std::vector<cv::Mat>& image) {
    auto res = det->run(all_arrays);
    return res;
  }

 private:
  std::string input_name;
  std::string kernel_name_1;
  std::string kernel_name_2;
  size_t batch_size;
  std::unique_ptr<X_Autonomous3D> det;
  std::vector<std::vector<float>> all_arrays;
};

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  std::string input = argv[3];
  std::string model_1 = argv[1];
  std::string model_2 = argv[2];
  std::string input_file_name = "./sample_x_autonomous3d.bin";
  if (input.find_last_of('/') != std::string::npos) {
    input_file_name =
        input.substr(0, input.find_last_of('/') + 1) + input_file_name;
  }

  return vitis::ai::main_for_performance(
      argc, argv, [input_file_name, model_1, model_2] {
        {
          return std::unique_ptr<vitis::ai::X_Autonomous3DPerf>(
              new vitis::ai::X_Autonomous3DPerf(input_file_name, model_1,
                                                model_2));
        }
      });
}
