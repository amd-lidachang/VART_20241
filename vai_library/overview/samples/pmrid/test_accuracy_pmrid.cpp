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
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/pmrid.hpp>
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

// parse raw_info from txt file, first para is raw fileneme and second para is
// ISO for every line.
static std::vector<std::pair<std::string, float>> parse_raw_info(
    const std::string& filename) {
  auto ret = std::vector<std::pair<std::string, float>>{};
  std::ifstream fin;
  fin.open(filename, std::ios_base::in);
  if (!fin) {
    std::cout << "Can't open the file " << filename << "\n";
    exit(-1);
  }
  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string name = "";  // raw fileneme
    int iso = 0;  // ISO parameter, KSigma transform need this parameter
    ss >> name >> iso;
    ret.push_back(std::make_pair(name, (float)iso));
  }
  fin.close();
  // for (const auto& r : ret) {
  //   LOG(INFO) << "input filename : " << r.first << "   ISO : " << r.second;
  // }
  return ret;
}

void create_dir(const std::string& path) {
  auto ret = mkdir(path.c_str(), 0755);
  if (ret == -1 && errno == ENOENT) {
    std::string path1 = path.substr(0, path.find_last_of('/'));
    create_dir(path1);
    create_dir(path);
    return;
  }
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
    std::cout << "error occured when mkdir " << path << "   " << errno
              << std::endl;
    exit(-1);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    cerr << "usage: " << argv[0]
         << " <model name> <RGB image path> <HHA image path> <image name file> "
            "<result_path>"
         << endl;
    return -1;
  }
  auto runner = vitis::ai::PMRID::create(argv[1], true);
  if (!runner) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto raw_infos = parse_raw_info(argv[2]);
  for (size_t i = 0; i < raw_infos.size(); i += runner->get_input_batch()) {
    std::vector<cv::Mat> imgs;
    std::vector<float> isos;
    for (size_t j = 0;
         j < runner->get_input_batch() && i + j < raw_infos.size(); j++) {
      cv::Mat img;
      img.create(3000, 4000, CV_16UC1);
      auto mode = std::ios_base::in | std::ios_base::binary;
      CHECK(std::ifstream(raw_infos[i + j].first, mode)
                .read((char*)img.data, 24000000)
                .good())
          << "fail to read! filename=" << raw_infos[i + j].first;

      imgs.push_back(img);
      isos.push_back(raw_infos[i + j].second);
    }
    auto results = runner->run(imgs, isos);
    for (size_t j = 0; j < imgs.size(); j++) {
      auto output_file = argv[3] + raw_infos[i + j].first + std::string(".out");
      string dir(output_file.begin(),
                 output_file.begin() + output_file.size() - 13);

      create_dir(dir);
      CHECK(std::ofstream(output_file, std::ios_base::out |
                                           std::ios_base::binary |
                                           std::ios_base::trunc)
                .write((char*)results[j].data(),
                       sizeof(float) * results[j].size())
                .good())
          << " faild to write to " << output_file;
    }
  }

  return 0;
}
