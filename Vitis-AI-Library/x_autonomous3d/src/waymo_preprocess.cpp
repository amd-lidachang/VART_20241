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
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/env_config.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include "./waymo_preprocess.hpp"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
DEF_ENV_PARAM(DEBUG_CENTERPOINT_WAYMO, "0");
DEF_ENV_PARAM(DEBUG_VOXEL, "0");
DEF_ENV_PARAM(DEBUG_SAVE, "0");

namespace vitis {
namespace ai {
namespace centerpoint_waymo {

int atomicAdd_logic(int *address, int val) {
   int old = *address;
   *address += val;
   return old;
}

void write_bin(const std::string& name, const int *input, int size) {
  std::ofstream out(name, std::ios::out |std::ios::binary);
  out.write((const char *)input, size * 4);
  out.close();
}


void write_bin(const std::string& name, const float *input, int size) {
  std::ofstream out(name, std::ios::out |std::ios::binary);
  out.write((const char *)input, size * 4);
  out.close();
}

void save_vector(const std::string& name, const float *input, int dim, int line) {
  std::ofstream out(name); 
  for (int i = 0; i < line; ++i) {
    for (int j = 0; j < dim; ++j) {
      out << input[i * dim + j] << " ";
    }
    out << '\n';
  }
  out.close();

}
void save_vector(const std::string& name, const int *input, int dim, int line) {
  std::ofstream out(name); 
  for (int i = 0; i < line; ++i) {
    for (int j = 0; j < dim; ++j) {
      out << input[i * dim + j] << " ";
    }
    out << '\n';
  }
  out.close();

}
void make_pillar_histo_kernel(
		const float* dev_points, const int in_num_points, 
                float* dev_pillar_point_feature_in_coors,
		int* pillar_count_histo, const int num_points,
		const int max_points_per_pillar, const int grid_x_size,
		const int grid_y_size, const int grid_z_size, const float min_x_range,
		const float min_y_range, const float min_z_range, const float pillar_x_size,
		const float pillar_y_size, const float pillar_z_size,
		const int num_point_feature) {
    int all_count = 0;
    LOG_IF(INFO, ENV_PARAM(DEBUG_CENTERPOINT_WAYMO)) 
          << "in_num_points:" << in_num_points; 
    for (auto point_i = 0; point_i < in_num_points; ++point_i) {
	int y_coor = std::floor((dev_points[point_i * num_point_feature + 1] - min_y_range) /
			pillar_y_size);
	int x_coor = std::floor((dev_points[point_i * num_point_feature + 0] - min_x_range) /
			pillar_x_size);
	int z_coor = std::floor((dev_points[point_i * num_point_feature + 2] - min_z_range) /
			pillar_z_size);
        if (x_coor >= 0 && x_coor < grid_x_size && y_coor >= 0 &&
			y_coor < grid_y_size && z_coor >= 0 && z_coor < grid_z_size) {
		//int count =
	        //		atomicAdd(&pillar_count_histo[y_coor * grid_x_size + x_coor], 1);
	        all_count++;
	        int count = pillar_count_histo[y_coor * grid_x_size + x_coor]++;
		if (count < max_points_per_pillar) {
			int ind =
				y_coor * grid_x_size * max_points_per_pillar * num_point_feature +
				x_coor * max_points_per_pillar * num_point_feature +
				count * num_point_feature;
			for (int i = 0; i < num_point_feature; ++i) {
				dev_pillar_point_feature_in_coors[ind + i] =
					dev_points[point_i * num_point_feature + i];
			}
		}
	//} else {
        //  LOG(INFO) << "skip x_coors:" << x_coor << " y_coors:" << y_coor << " z_coors:" << z_coor 
        //            << " val x:" << dev_points[point_i * num_point_feature + 0]
        //            << " val y:" << dev_points[point_i * num_point_feature + 1]
        //            << " val z:" << dev_points[point_i * num_point_feature + 2];
        }
    }
    
    LOG_IF(INFO, ENV_PARAM(DEBUG_CENTERPOINT_WAYMO)) 
          << "all count:" << all_count; 
}


void make_pillar_index_kernel(
		int* dev_pillar_count_histo, int* dev_counter, int* dev_pillar_count,
		int* dev_x_coors, int* dev_y_coors, int* dev_num_points_per_pillar,
		const int max_pillars,
		const int max_points_per_pillar, 
                const int grid_x_size, const int grid_y_size) {
    
    for (int y = 0; y < grid_y_size; y++) {
        for (int x = 0; x < grid_x_size; x++) {
	    int num_points_at_this_pillar = dev_pillar_count_histo[y * grid_x_size + x];
	    if (num_points_at_this_pillar == 0) {
		continue;
	    }
	    //int count = atomicAdd(dev_counter, 1);
	    int count = (*dev_counter)++;
	    if (count < max_pillars) {
		//atomicAdd(dev_pillar_count, 1);
		(*dev_pillar_count)++;

		if (num_points_at_this_pillar >= max_points_per_pillar) {
			dev_num_points_per_pillar[count] = max_points_per_pillar;
		} else {
			dev_num_points_per_pillar[count] = num_points_at_this_pillar;
		}
		dev_x_coors[count] = x;
		dev_y_coors[count] = y;
            }
	}
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_CENTERPOINT_WAYMO))
          << "count : " << *dev_counter;    
}

void make_pillar_feature_kernel(
		float* dev_pillar_point_feature_in_coors, float* dev_pillar_point_feature,
		int * dev_pillar_coors, int* dev_x_coors, int* dev_y_coors,
		int* dev_num_points_per_pillar, 
                const int pillar_count, const int max_points, 
		const int num_point_feature, const int grid_x_size) {
    for (auto ith_pillar = 0; ith_pillar < pillar_count; ++ith_pillar) { 
	int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
        for (auto ith_point = 0; ith_point < num_points_at_this_pillar; ++ith_point) {
	    int x_ind = dev_x_coors[ith_pillar];
	    int y_ind = dev_y_coors[ith_pillar];
	    int pillar_ind = ith_pillar * max_points * num_point_feature +
		ith_point * num_point_feature;
	    int coors_ind = y_ind * grid_x_size * max_points * num_point_feature +
		x_ind * max_points * num_point_feature +
		ith_point * num_point_feature;
	    for (int i = 0; i < num_point_feature; ++i) {
		dev_pillar_point_feature[pillar_ind + i] =
			dev_pillar_point_feature_in_coors[coors_ind + i];
            }
            
	    float coor_x = static_cast<float>(x_ind);
	    float coor_y = static_cast<float>(y_ind);
	    dev_pillar_coors[ith_pillar * 4 + 0] = 0;  // batch idx
	    dev_pillar_coors[ith_pillar * 4 + 1] = 0;  // z
	    dev_pillar_coors[ith_pillar * 4 + 2] = coor_y;
	    dev_pillar_coors[ith_pillar * 4 + 3] = coor_x;
        }
    } 
}
	

void pillar_mean_kernel(
	float* dev_points_mean, 
	const int num_point_feature,
	const float* dev_pillar_point_feature, 
	const int* dev_num_points_per_pillar, 
	//int max_pillars,
	int pillar_count, 
	int max_points_pre_pillar) {
    for (auto ith_pillar = 0; ith_pillar < pillar_count; ++ith_pillar) {
	int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
        vector<float> points_sum(3, 0.f);
        for (auto ith_point = 0; ith_point < num_points_at_this_pillar; ++ith_point) {
            for (int i = 0; i < 3; ++i) {
                points_sum[i] += dev_pillar_point_feature[ith_pillar * max_points_pre_pillar * num_point_feature + ith_point * num_point_feature + i];
            }
            
        }        
        for (int i = 0; i < 3; ++i) {
            dev_points_mean[ith_pillar * 3 + i] = points_sum[i] / num_points_at_this_pillar;
        }
    }  
}

void gather_point_feature_kernel(
	const int pillar_count,const int max_num_points_per_pillar, const int num_point_feature,
	const float min_x_range, const float min_y_range, const float min_z_range, 
	const float pillar_x_size,  const float pillar_y_size, const float pillar_z_size,
	const float* dev_pillar_point_feature, const int* dev_num_points_per_pillar, 
	const int* dev_pillar_coors,
	float* dev_points_mean, 
	float* dev_pfn_gather_feature){

  int num_gather_feature = 11;
  for (auto ith_pillar = 0; ith_pillar < pillar_count; ith_pillar++) {
    int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
    for (auto ith_point = 0; ith_point < num_points_at_this_pillar; ++ith_point) {
      int feat_ind = ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature;
      int pfe_ind = ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature;
      // gather[:, :, 0:6] = voxel_features[:, :, 0:6]
      for (int i = 0; i < num_point_feature; ++i) {
      	dev_pfn_gather_feature[pfe_ind + i] = dev_pillar_point_feature[feat_ind + i];
      }
      // gather[:, :, 6:9] = voxel_features[:, :, :3] - points_mean
      dev_pfn_gather_feature[pfe_ind + num_point_feature + 0] = dev_pillar_point_feature[feat_ind + 0] - dev_points_mean[ith_pillar * 3 + 0];
      dev_pfn_gather_feature[pfe_ind + num_point_feature + 1] = dev_pillar_point_feature[feat_ind + 1] - dev_points_mean[ith_pillar * 3 + 1];
      dev_pfn_gather_feature[pfe_ind + num_point_feature + 2] = dev_pillar_point_feature[feat_ind + 2] - dev_points_mean[ith_pillar * 3 + 2];
 
      // gather[:, :, 9] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
      dev_pfn_gather_feature[pfe_ind + num_point_feature + 3]  
	=  dev_pillar_point_feature[feat_ind + 0] - (dev_pillar_coors[ith_pillar * 4 + 3] * pillar_x_size + (pillar_x_size/2 + min_x_range));

      // gather[:, :, 10] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
      dev_pfn_gather_feature[pfe_ind + num_point_feature + 4]  
	=  dev_pillar_point_feature[feat_ind + 1] - (dev_pillar_coors[ith_pillar * 4 + 2] * pillar_y_size + (pillar_y_size/2 + min_y_range));
    }
  }  
}

void generate_pillars_core(const float* dev_points, 
		const int in_num_points, const int max_num_pillars, const int max_num_points_per_pillar, const int num_point_feature,
		const int grid_x_size, const int grid_y_size, const int grid_z_size,
		const float min_x_range, const float min_y_range, const float min_z_range,
		const float pillar_x_size, const float pillar_y_size, const float pillar_z_size,
		float*  dev_pillar_point_feature_in_coors,
		int* dev_pillar_count_histo, int* dev_counter, int* dev_pillar_count,
		int* dev_x_coors, int* dev_y_coors, 
		int* dev_num_points_per_pillar,
		float* dev_pillar_point_feature, int* dev_pillar_coors,
		float* dev_points_mean,
		float* dev_pfn_gather_feature
		) {

    // 1. move points (n, 6) to pillsr_points_features_in_coors (624, 624, 20, 6) and record pillar count(points num in a pillar, not more than 20) in pillar_count_histo (624, 624)
__TIC__(MAKE_PILLAR_HISTO_KERNEL)
   make_pillar_histo_kernel(
			dev_points, in_num_points,
                        dev_pillar_point_feature_in_coors, dev_pillar_count_histo,
			in_num_points, max_num_points_per_pillar, grid_x_size, grid_y_size,
			grid_z_size, min_x_range, min_y_range, min_z_range, pillar_x_size,
			pillar_y_size, pillar_z_size, num_point_feature);
__TOC__(MAKE_PILLAR_HISTO_KERNEL)
       
    //if(ENV_PARAM(DEBUG_SAVE)) {
    //  save_vector("pillar_count_histo.txt", dev_pillar_count_histo, 624, 624);
    //  for (auto i = 0; i < 624; ++i) {
    //    int sum = 0;
    //    for (auto j = 0; j < 624; ++j) {
    //      sum += dev_pillar_count_histo[i * 624 + j];
    //    }
    //    std::cout << "histo " << i << ":" << sum << std::endl; 
    //  }
    //}
    // 2. record pillar number in couter (all) and pillar_count (valid, not more than max_pillars)
    //    record points num of every pillar in num_points_per_pillar (60000)
    //    record coors(x,y) in x_coors(60000) and y_coors(60000)
__TIC__(MAKE_PILLAR_INDEX_KERNEL)
    make_pillar_index_kernel(
			dev_pillar_count_histo, dev_counter, dev_pillar_count, dev_x_coors,
			dev_y_coors, dev_num_points_per_pillar,
			max_num_pillars, max_num_points_per_pillar, grid_x_size, grid_y_size);
__TOC__(MAKE_PILLAR_INDEX_KERNEL)

    // 3. copy points from point_feature_in_coors(624, 624, 20, 6) to pillar_point_feature (60000, 20, 6) 
    //    record pillar_coors (60000, 4) 
    //    pillar_coors last dim [0, 0, coor_y, coor_x]
    
__TIC__(MAKE_PILLAR_FEATURE_KERNEL)
    make_pillar_feature_kernel(
			dev_pillar_point_feature_in_coors, dev_pillar_point_feature,
			dev_pillar_coors, dev_x_coors, dev_y_coors, dev_num_points_per_pillar,
			*dev_pillar_count, max_num_points_per_pillar, num_point_feature, grid_x_size);
__TOC__(MAKE_PILLAR_FEATURE_KERNEL)

    // 4. calculate points mean in pillar_point_mean
__TIC__(MAKE_PILLAR_MEAN_KERNEL)
    pillar_mean_kernel(
        dev_points_mean, num_point_feature, dev_pillar_point_feature, dev_num_points_per_pillar, 
	*dev_pillar_count, max_num_points_per_pillar);
__TOC__(MAKE_PILLAR_MEAN_KERNEL)

    // 5. gather features: from (60000, 20, 6) to (60000, 20, 11)
    // last dim 6 to 11
    // gather[6:9] = voxel[:3] - mean 
    // gather[9] = voxel[0] - (coors_x * pillar_x_size + pillar_x_size/2 + min_x_range) // center_x 
    // gather[10] = voxel[1] - (coors_y * pillar_y_size + pillar_y_size/2 + min_y_range) // center_y
__TIC__(GATHER_POINT_FEATURE_KERNEL)
    gather_point_feature_kernel(
      *dev_pillar_count, max_num_points_per_pillar, num_point_feature,
      min_x_range, min_y_range, min_z_range,
      pillar_x_size, pillar_y_size, pillar_z_size, 
      dev_pillar_point_feature, dev_num_points_per_pillar, dev_pillar_coors,
      dev_points_mean,
      dev_pfn_gather_feature);
__TOC__(GATHER_POINT_FEATURE_KERNEL)

}

void generate_pillars(const vector<float> &points, // [n, 6]
    // in_num_points : n,  max_num_pillars: 60000, max_num_points_per_pillar: 20, num_point_feature: 6
    const int in_num_points, const int max_num_pillars, const int max_num_points_per_pillar, const int num_point_feature,
    const int grid_x_size, const int grid_y_size, const int grid_z_size,
    // [-74.88, -74.88, -2]
    const float min_x_range, const float min_y_range, const float min_z_range,
    // voxel_size [0.24, 0.24, 6]
    const float pillar_x_size, const float pillar_y_size, const float pillar_z_size,
    // [624, 624, 20, 6]
    vector<float> &pillar_point_feature_in_coors,
    // [624, 624]
    vector<int> &pillar_count_histo, 
    int &counter, int &pillar_count, // output num
    // 60000, 60000
    vector<int> &x_coors, vector<int> &y_coors,
    // 60000 : output, [60000, 20, 6], [60000, 4]: output 
    vector<int> &num_points_per_pillar, vector<float> &pillar_point_feature, vector<int> &pillar_coors,
    // [60000, 1, 3]
    vector<float> &pillar_point_mean,
    // [60000, 20, 11] : output
    vector<float> &pillar_pfn_feature) {

__TIC__(CENTERPOINT_VOXELIZE)
  generate_pillars_core(points.data(),
		in_num_points, max_num_pillars, max_num_points_per_pillar, num_point_feature,
		grid_x_size, grid_y_size, grid_z_size, 
		min_x_range, min_y_range, min_z_range,
		pillar_x_size, pillar_y_size, pillar_z_size,
		pillar_point_feature_in_coors.data(),        
                pillar_count_histo.data(),
		&counter, &pillar_count, 
                x_coors.data(), y_coors.data(),
		num_points_per_pillar.data(), pillar_point_feature.data(), pillar_coors.data(),
		pillar_point_mean.data(),
		pillar_pfn_feature.data());
__TOC__(CENTERPOINT_VOXELIZE)
 
}

void normalize_set_input(const vector<float> &feature, const vector<float> scales, float tensor_scale, int8_t *input_tensor, int tensor_size) {
  std::memset(input_tensor, 0, tensor_size);
  int valid_size = std::min((int)feature.size(), tensor_size);
  int dim = scales.size();
  int len = valid_size / dim; 
  for (int i = 0; i < len; ++i) {
    for (int j = 0; j < dim; ++j) {
      input_tensor[i * dim + j] = (int)(feature[i * dim + j] / scales[j] * tensor_scale);
    }
  }
}

void normalize(const vector<float> &feature, const vector<float> scales, vector<float> &output) {
  int valid_size = feature.size(); 
  int dim = scales.size();
  int len = valid_size / dim; 
  for (int i = 0; i < len; ++i) {
    for (int j = 0; j < dim; ++j) {
      output[i * dim + j] = (feature[i * dim + j] / scales[j]);
    }
  }
}



void set_input(const vector<float> &feature, float scale, int8_t *input_tensor, int tensor_size) {
  std::memset(input_tensor, 0, tensor_size);
  int valid_size = std::min((int)feature.size(), tensor_size);
  for (int i = 0; i < valid_size; ++i) {
    input_tensor[i] = (int)(feature[i] * scale);
  }
}

void voxel_reorder(vector<float> &feature, int pillar_count, vector<int> &coors, vector<int> &num_per_pillar, const vector<int> &normal_coors){
  int voxel_length = 20 * 11;
  LOG(INFO) << "feature size:" << feature.size();
  LOG(INFO) << "coors size:" << coors.size();
  LOG(INFO) << "num_per_pillar size:" << num_per_pillar.size();
  LOG(INFO) << "normal_coors size:" << normal_coors.size();
  //int coors_dim = 4;
  vector<int> coors_map(624 * 624);
  for (int i = 0; i < pillar_count; ++i) {
    int x = coors[i * 4 + 3];
    int y = coors[i * 4 + 2];
    coors_map[y * 624 + x] = i;
  }
  vector<float> reordered_feature(voxel_length * pillar_count);
  vector<int> reordered_num_per_pillar(pillar_count);
  for (int i = 0; i < pillar_count; ++i) {
    auto x = normal_coors[i * 4 + 3];
    auto y = normal_coors[i * 4 + 2];
    auto index = coors_map[y * 624 + x];
    if (i < 10) {
      LOG(INFO) << "original index:" << index
                << " normal index:" << i;
    }
    //std::memcpy(reordered_feature.data() + i * voxel_length * 4, 
    //            feature.data() + index * voxel_length * 4,
    //            voxel_length * 4);
    for (int k = 0; k < voxel_length; ++k) {
      reordered_feature[i * voxel_length + k] = feature[index * voxel_length + k];
    }
    reordered_num_per_pillar[i] = num_per_pillar[index];
  }
  std::copy(reordered_feature.begin(), reordered_feature.end(), feature.begin());
  std::copy(normal_coors.begin(), normal_coors.end(), coors.begin());
  std::copy(reordered_num_per_pillar.begin(), reordered_num_per_pillar.end(), num_per_pillar.begin());
  //feature = reordered_feature;
  //coors = normal_coors;
  //num_per_pillar = reordered_num_per_pillar;
}

}}}
