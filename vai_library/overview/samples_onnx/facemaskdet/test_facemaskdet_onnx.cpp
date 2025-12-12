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

#include "facemaskdet_onnx.hpp"

using namespace cv;
int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << "  <pic_url> [pic_url]... " << std::endl;
    abort();
  }
  auto model_name = std::string(argv[1]);
  auto det = OnnxFacemaskdet::create(model_name);

  std::vector<Mat> vimg;
  for(int i=2; i<argc; i++) {
    Mat img = imread(argv[i]);
    if (img.empty()) {
      cerr << "can't load image! " << argv[i] << endl;
      return -1;
    }
    vimg.push_back(img);
  } 
  auto result = det->run(vimg);

  for(int i=0; i<(int)result.size(); i++){
     std::cout <<"batch " << i <<"\n";    
     for (auto &box : result[i].bboxes) {
        int label = box.label;
        float xmin = box.x * vimg[i].cols + 1;
        float ymin = box.y * vimg[i].rows + 1;
        float xmax = xmin + box.width * vimg[i].cols;
        float ymax = ymin + box.height * vimg[i].rows;
        if (xmin < 0.) xmin = 1.;
        if (ymin < 0.) ymin = 1.;
        if (xmax > vimg[i].cols) xmax = vimg[i].cols;
        if (ymax > vimg[i].rows) ymax = vimg[i].rows;
        float confidence = box.score;
    
        cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t" << xmax
             << "\t" << ymax << "\t" << confidence << "\n";
     }
  }       
  return 0;
}

