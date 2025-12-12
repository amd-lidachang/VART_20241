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
#include <memory>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/centernet.hpp>

using namespace std;
using namespace cv;
namespace fs=std::filesystem;

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage :" << argv[0] << " <model_name> <test_dir>  <result_file>" 
              << "\n        model_name is centernet_pt " << std::endl;
    abort();
  }

  auto net = vitis::ai::CenterNet::create(argv[1]);

  std::string test_dir(argv[2]);  

  ofstream Tout;
  Tout.open(argv[3], ios_base::out);
  if(!Tout) {
    cout<<"Can't open the file! " << argv[3] << "\n";
    return -1;
  }

  std::vector<fs::path> vpath;
  for(auto it=fs::directory_iterator(test_dir); it != fs::directory_iterator(); ++it) {
    if (fs::is_regular_file(*it)) {
       vpath.emplace_back(it->path() );
    }
  }
  std::sort(vpath.begin(), vpath.end());
  /*
  struct Det_Res{
    float score;
    std::array<float, 4> pos;
    Det_Res(float score1, std::array<float,4> pos1) : score(score1), pos(pos1){ }
  };
  std::vector<std::vector<Det_Res>> vres;
  */
  /* output file  format :
     file_name_which_is_int_number
     class_idx amount_for_this_id
     float0 f1 f2 f3 score
     float0 f1 f2 f3 score
     .....
   */
  for(auto& it : vpath) {
    Mat img = cv::imread(it.string());
    if (img.empty()) {
      cerr << "cannot load " << it << endl;
      exit(-1);
    }
    auto r= net->run(img);  
    Tout <<  atoi(it.stem().string().c_str()) <<"\n";
    for(int i=0; i<80; i++) {
      if(r.vres[i].empty()) {
         Tout << i+1 << " " << 0 << "\n";
      } else {
         Tout << i+1 << " " << r.vres[i].size() << "\n";
         // for(int j=0; j<(int)r.vres[i].size(); j++) {
         for(int j=(int)r.vres[i].size()-1; j>=0; --j) {
            for(int k=0; k<4; k++) {
               Tout << r.vres[i][j].pos[k] << " " ;
            }
            Tout << r.vres[i][j].score <<"\n";
         }
      }
    }
  }

  Tout.close();
  return 0;
}

