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
#include <thread>
#include <vector>
#include <numeric>
#include <sys/stat.h>
#include <stdlib.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vitis/ai/bosch_ssdobjdet.hpp>

using namespace cv;
using namespace std;
namespace fs=std::filesystem;
using namespace vitis::ai;

std::string out_fname;
std::string db_name;
std::vector<int> picnum;
vector<string> names;

void accuracy_thread(Bosch_Ssdobjdet* det, int i, int t_n, std::string f_name);

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
  if (argc < 5) {
    std::cerr << "usage : " << argv[0] << " model_name  <video_list_file> <db_dir> <output_file> [thread_num]" << std::endl
              << "          model_name is bosch_ssdobjdet " << std::endl;
    abort();
  }


  std::string model = argv[1];
  std::string list_name(argv[2]);
  db_name = std::string(argv[3]);
  out_fname = std::string(argv[4]);

  LoadListNames(list_name, names);

  int t_n = 1;
  if (argc==6) {
    t_n = atoi(argv[5]);
  }

#if 0   // format as below :
[
 {
  'image_id': 44,
  'category_id': 1,  // no value is 0
  'bbox': [668.808, 744.6, 25.803, 58.555],
  'score': 0.5358808040618896,
  'height': 58.555
 },
 {}
]
#endif

  picnum.assign(names.size()+1, 0);
  for(int j=0; j<(int)names.size(); j++) {
    std::string f = names[j];
    std::string cur_dir = db_name + "/" + f;
    for(auto it=fs::directory_iterator(cur_dir); it != fs::directory_iterator(); ++it) {
      if (fs::is_regular_file(*it)) {
         picnum[j+1]++;
      }
    }
    if (j>0) { picnum[j] += picnum[j-1]; }
    // std::cout <<"picnum : " << j << " " << picnum[j] <<"\n";
  }

  std::vector<std::thread> vth;
  std::vector< std::unique_ptr<Bosch_Ssdobjdet>> vssd;
  std::vector<std::string> out_fname_t(t_n);
  for(int i=0; i<t_n; i++) {
    vssd.emplace_back(vitis::ai::Bosch_Ssdobjdet::create(model));
    out_fname_t[i] = out_fname + "_" + std::to_string(i);
    // std::cout <<" out_fname :" << out_fname_t[i]  << "\n";
    vth.emplace_back( std::thread( &accuracy_thread, vssd[i].get(), i , t_n, out_fname_t[i]));
  }
  for(int i=0; i<t_n; i++) {
    vth[i].join();
  }

  // combine output files;
  std::string cmd("cat ");
  for(int i=0; i<t_n; i++) {
     cmd.append(out_fname_t[i]);
     cmd.append(" ");
  }
  cmd.append(" > ");
  cmd.append(out_fname);
  auto ret = system(cmd.c_str());
  cmd = "rm -f ";
  cmd.append(out_fname);
  cmd.append("_*");
  ret = system(cmd.c_str());
  (void)ret;
  // std::cout <<"ret : " << ret << " OK\n";

  return 0;
}

void accuracy_thread(Bosch_Ssdobjdet* det, int i, int t_n, std::string f_name)
{
  std::ofstream out_fs(f_name, std::ofstream::out);
  if(!out_fs)  {
      std::cout<<"Can't open the file result!";
      abort();
  }

  if(i==0)  out_fs <<"[\n";

  int all_size = (int)names.size();
  // for(auto& f: names) {
  int split = all_size/t_n;
  int start = i*split;
  int end = ((i != t_n-1) ? split : all_size-i*split);
  int f_size = ((i != t_n-1) ?all_size : all_size-i*split);
  int image_id = picnum[i*split];
  // std::cout <<"image_id : " << i << " " << image_id << "   " << f_size  << "  " << start << " " << end <<"\n";

  for(int j=start; j<start+end; j++) {
    std::string f = names[j];
    f_size--;
    std::vector<fs::path> vpath;
    std::string cur_dir = db_name + "/" + f;
    for(auto it=fs::directory_iterator(cur_dir); it != fs::directory_iterator(); ++it) {
      if (fs::is_regular_file(*it)) {
         vpath.emplace_back(it->path() );
      }
    }
    std::sort(vpath.begin(), vpath.end());
    int it_size = (int)vpath.size();
    for(auto& it: vpath) {
       it_size--;
       Mat img = cv::imread(it.string());
       if (img.empty()) {
         std::cout << "cannot load " << it << "\n";
         exit(-1);
       }
       auto result = det->run(img);
       if (result.bboxes.size()) {
         int res_size = (int)result.bboxes.size();
         for(auto &r: result.bboxes) {
           res_size--;
           out_fs << "{ ";
           out_fs <<  "'image_id': " << image_id << ", "
                  <<  "'category_id': 1, "
                  <<  "'bbox': [ " << r.x << "," << r.y << "," << r.width << "," << r.height << "], "
                  <<  "'score': " << r.score << ", "
                  <<  "'height': " << r.height << " "
                  << "}" ;

           if (!(!f_size && !it_size && !res_size))  out_fs <<",\n";
         }
       } else {
           out_fs << "{ "
                  <<  "'image_id': " << image_id << ", "
                  <<  "'category_id': 0, "
                  <<  "'bbox': [0,0,0,0], "
                  <<  "'score': 0, "
                  <<  "'height': 0 "
                  << "}"  ;
           if (!(!f_size && !it_size ))  out_fs <<",\n";
       }
       image_id++;
    }
  }

  if(i == t_n-1)  out_fs <<"\n]";

  out_fs.close();
}

