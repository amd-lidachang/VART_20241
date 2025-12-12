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

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/bosch_ssdobjdet.hpp>

using namespace cv;
using namespace std;


int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage :" << argv[0] << " <model_name> <image_url> [<image_url> ...]"
              << std::endl;
    abort();
  }

  auto det =
      vitis::ai::Bosch_Ssdobjdet::create(argv[1]);
  if (!det) {
     std::cerr <<"create error for model " << argv[1] << "\n";
     abort();
  }
  std::vector<cv::Mat> arg_input_images;
  std::vector<cv::Size> arg_input_images_size;
  std::vector<std::string> arg_input_images_names;
  for (auto i = 2; i < argc; i++) {
    cv::Mat img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    arg_input_images.push_back(img);
    arg_input_images_size.push_back(img.size());
    arg_input_images_names.push_back(argv[i]);
  }

  if (arg_input_images.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
  }

  std::vector<cv::Mat> batch_images;
  std::vector<std::string> batch_images_names;
  std::vector<cv::Size> batch_images_size;
  auto batch = det->get_input_batch();
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(
        arg_input_images_names[batch_idx % arg_input_images.size()]);
    batch_images_size.push_back(
        arg_input_images_size[batch_idx % arg_input_images.size()]);
  }

  auto results = det->run(batch_images);

  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    float scale_x = batch_images[batch_idx].cols/(float)results[batch_idx].width;
    float scale_y = batch_images[batch_idx].rows/(float)results[batch_idx].height;

    std::cout <<"batch-" << batch_idx << ":\n";
    for(int i=0; i<(int) results[batch_idx].bboxes.size(); i++) {
      std::cout << results[batch_idx].bboxes[i].score << "  "
                << results[batch_idx].bboxes[i].x*scale_x<< " "
                << results[batch_idx].bboxes[i].y*scale_y<< " "
                << results[batch_idx].bboxes[i].width*scale_x<< " "
                << results[batch_idx].bboxes[i].height*scale_y<< "\n";
    }
  }
  return 0;
}

