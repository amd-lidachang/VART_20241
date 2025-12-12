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

#include <glog/logging.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/arflow.hpp>
using namespace std;
using namespace cv;

std::vector<std::string> split(const std::string& s, const std::string& delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0) return elems;
  while (pos < len) {
    int find_pos = s.find(delim, pos);
    if (find_pos < 0) {
      elems.push_back(s.substr(pos, len - pos));
      break;
    }
    elems.push_back(s.substr(pos, find_pos - pos));
    pos = find_pos + delim_len;
  }
  return elems;
}

void LoadImageNames(std::string const& filename,
                    std::vector<std::string>& images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE* fp = fopen(filename.c_str(), "r");
  if (NULL == fp) {
    fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);
  }

  char buffer[256] = {0};
  while (fgets(buffer, 256, fp) != NULL) {
    int n = strlen(buffer);
    buffer[n - 1] = '\0';
    std::string name = buffer;
    images.push_back(name);
  }

  fclose(fp);
}
std::vector<float> convert_fixpoint_to_float(
    const vitis::ai::library::OutputTensor& tensor, size_t j) {
  auto scale = vitis::ai::library::tensor_scale(tensor);
  auto data = (signed char*)tensor.get_data(j);
  auto size = tensor.width * tensor.height * tensor.channel;
  auto ret = std::vector<float>(size);
  transform(data, data + size, ret.begin(),
            [scale](signed char v) { return ((float)v) * scale; });
  return ret;
}
int main(int argc, char* argv[]) {
  if (argc != 4) {
    cerr << "usage: " << argv[0]
         << " <model name> <image name file> <result_path>" << endl;
    return -1;
  }
  auto arflow = vitis::ai::ARFlow::create(argv[1], true);
  if (!arflow) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  vector<string> names;
  LoadImageNames(argv[2], names);

  string g_output_dir = argv[3];
  {
    vector<string> output_dir{g_output_dir, g_output_dir + "/colored_0",
                              g_output_dir + "/image_2"};
    for (auto& dir : output_dir) {
      string mkdir = "mkdir -p " + dir;
      auto a = system(mkdir.c_str());
      if (a == -1) exit(0);
    }
  }
  auto batch = arflow->get_input_batch();
  auto mode = ios_base::out | ios_base::binary | ios_base::trunc;
  for (auto i = 0u; i < names.size(); i += 2 * batch) {
    vector<Mat> img1s, img2s;
    for (size_t j = 0; j < batch && i + j * 2 + 1 < names.size(); j++) {
      auto img1 = cv::imread(names[i + j * 2]);
      CHECK(!img1.empty()) << "cannot load " << names[i + j * 2];
      auto img2 = cv::imread(names[i + j * 2 + 1]);
      CHECK(!img2.empty()) << "cannot load " << names[i + j * 2 + 1];
      img1s.emplace_back(img1);
      img2s.emplace_back(img2);
    }
    auto result = arflow->run(img1s, img2s)[0];
    for (size_t j = 0; j < img1s.size(); j++) {
      auto data = convert_fixpoint_to_float(result, j);
      string dir = g_output_dir;
      if (names[i + j * 2].find("colored_0") != std::string::npos) {
        dir = g_output_dir + "/colored_0/";

      } else {
        dir = g_output_dir + "/image_2/";
      }
      ofstream(
          dir + split(*split(names[i + j * 2], "/").rbegin(), ".")[0] + ".bin",
          mode)
          .write((char*)data.data(), data.size() * sizeof(float));
    }
  }
  return 0;
}
