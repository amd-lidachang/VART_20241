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
#include <sys/stat.h>
#include "sesrs_onnx.hpp"

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

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>" << " <list_file1> <outdir> " << std::endl;
    abort();
  }

  vector<string> vlist;
  LoadListNames(argv[2], vlist);
   
  auto ret = mkdir(argv[3], 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
     std::cout << "error occured when mkdir " << argv[3] << std::endl;
     return -1;
  }

  auto net = OnnxSesrs::create(argv[1]);
  int batch = net->get_input_batch();
  std::vector<cv::Mat> vm;
  for(int i=0, j=0; i<(int)vlist.size(); ) {
    vm.clear();
    for(j=0; j<batch && i<(int)vlist.size(); j++, i++) {
      Mat img = cv::imread(vlist[i]);
      if (img.empty()) {
        cerr << "cannot load " << vlist[i] << endl;
        abort();
      }
      vm.emplace_back(img);
    }
    auto ret = net->run(vm);
    for(int k=0; k<j; k++) {
      std::string filen = std::string(argv[3]) + vlist[i-j+k].substr(vlist[i-j+k].find_last_of('/') );
      cv::imwrite(filen, ret[k].mat); 
    }
  }
  return 0;
}


