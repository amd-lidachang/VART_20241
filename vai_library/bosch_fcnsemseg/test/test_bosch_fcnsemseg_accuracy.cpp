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

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <thread>
#include <sys/stat.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vitis/ai/bosch_fcnsemseg.hpp>

using namespace cv;
using namespace std;
using namespace vitis::ai;

namespace fs=std::filesystem;

vector<string> names;
std::string db_name;
std::string outdir_name;

void LoadListNames(const std::string& filename,  std::vector<std::string> &vlist)
{
  ifstream Tin;
  Tin.open(filename, ios_base::in);
  std::string str;
  if(!Tin)  {
     std::cout<<"Can't open the file " << filename << "\n";      exit(-1);
  }
  while( getline(Tin, str)) {
    vlist.emplace_back(str);
  }
  Tin.close();
}

void accuracy_thread(Bosch_Fcnsemseg* seg, int i, int t_n){
    for(int j=i; j<(int)names.size(); j+=t_n){
      auto f = names[j];
      std::string pic_path = db_name+"/"+f;
      Mat img = cv::imread(pic_path);
      if (img.empty()) {
        std::cout << "cannot load " << pic_path << "\n";
        exit(-1);
      }
      auto result = seg->run(img);
      std::string fname_main=f.substr(0, f.size()-4);
      fname_main = outdir_name + "/" + fname_main+".png";
      imwrite(fname_main, result.segmentation );
    }
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cerr << "usage : " << argv[0] << " model_name  <video_list_file> <db_dir> <output_dir>  [thread_num]" << std::endl
              << "          model_name is bosch_fcnsemseg " << std::endl;
    abort();
  }

  std::string model = argv[1];

  std::string list_name(argv[2]);
  db_name = std::string(argv[3]);
  outdir_name = std::string(argv[4]);

  auto ret = mkdir(outdir_name.c_str(), 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
    std::cout << "error occured when mkdir " << outdir_name << std::endl;
    return -1;
  }

  LoadListNames(list_name, names);

  int t_n = 1;
  if (argc==6) {
    t_n = atoi(argv[5]);
  }

  std::vector<std::thread> vth;
  std::vector< std::unique_ptr<Bosch_Fcnsemseg>> vseg;
  if (t_n>0) {
    for(int i=0; i<t_n; i++) {
      vseg.emplace_back(vitis::ai::Bosch_Fcnsemseg::create(model));
      vth.emplace_back( std::thread( &accuracy_thread, vseg[i].get(), i , t_n));
    }
    for(int i=0; i<t_n; i++) {
      vth[i].join();
    }
  }
  return 0;
}


