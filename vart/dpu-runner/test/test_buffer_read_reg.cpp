#include <glog/logging.h>
#include <xrt/xrt/xrt_device.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <cstdlib>
#include <unistd.h>

#include <xir/tensor/tensor.hpp>
#include "vart/dpu/vitis_dpu_runner_factory.hpp"
#include "vart/runner_ext.hpp"
#include "xir/xrt_device_handle.hpp"
#include "xir/graph/graph.hpp"

using namespace std;




void run_inference(const string& xmodel, const string& kernel,
                   const string& input_file, int runner_num, int count,
                   const string& xclbin_path) {

  std::ifstream stream(xclbin_path, std::ios::binary);

  LOG(INFO) << "Loaded xclbin: " << xclbin_path;

  // Create DPU runners
  setenv("XLNX_VART_FIRMWARE", xclbin_path.c_str(), 1);
  vector<std::unique_ptr<vart::Runner>> runners;

     auto graph = xir::Graph::deserialize(xmodel);
     auto root = graph->get_root_subgraph();
     xir::Subgraph* dpu_subgraph = nullptr;
        for (auto c : root->children_topological_sort()) {
        if (c->get_attr<std::string>("device") == "DPU") {
            dpu_subgraph = c;
            break;
        }
    }
        if (!dpu_subgraph) {
        std::cerr << "No DPU subgraph found in the model!" << std::endl;
        return ;
    }
      auto attrs = xir::Attrs::create();
    runners.emplace_back(vart::RunnerExt::create_runner(dpu_subgraph, attrs.get()));
 
  for (int i = 0; i < runner_num; ++i) {
    auto& runner = runners[i];
    auto r = dynamic_cast<vart::RunnerExt*>(runner.get());
    auto input = r->get_inputs();
    CHECK_EQ(input.size(), 1u) << "Only support single input.";
    auto output = r->get_outputs();
    
    size_t batch = input[0]->get_tensor()->get_shape()[0];
    size_t size_per_batch = input[0]->get_tensor()->get_data_size() / batch;

    for (size_t b = 0; b < batch; ++b) {
      uint64_t input_data = 0;
      uint64_t input_size = 0;
      std::tie(input_data, input_size) = input[0]->data({(int)b, 0, 0, 0});
      CHECK(std::ifstream(input_file).read((char*)input_data, size_per_batch).good())
          << "Failed to read input";
    }

    for (int c = 0; c < count; ++c) {
      for (auto& in : input) {
        in->sync_for_write(0, in->get_tensor()->get_data_size() / batch);
      }
      for (int i=0; i<10; i++){
      LOG(INFO)<<"interation run "<<i;
      runner->execute_async(input, output);
      runner->wait(0, 0);
      }
      for (auto& out : output) {
        out->sync_for_read(0, out->get_tensor()->get_data_size() / batch);
      }

    }
    // sleep(1);
    // auto h= xir::XrtDeviceHandle::get_instance();
    // h.reset();
    // h.reset();
  }




  runners.clear(); 
  // sleep(1);
  // auto dpu = xir::DpuController::get_instance();


}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <xmodel_path> <kernel_name> <input_file> <runner_num> <batch_count>\n";
    return 1;
  }

  const string xmodel = argv[1];
  const string kernel = argv[2];
  const string input_file = argv[3];
  const int runner_num = std::stoi(argv[4]);
  const int count = std::stoi(argv[5]);

  const string xclbin1 = "/usr/lib/rp.xclbin";
  const string xclbin2 = "/usr/lib/rm.xclbin";

  for (int iter = 0; iter < 1000; ++iter) {
    LOG(INFO)<<"============================Read the Reg 0xa4200000=============================";
    system("devmem 0xa4200000");
    LOG(INFO)<<"======================Write the Reg 0xA4200014==============================";
    system("devmem 0xA4200014 32 0x00000001");
    LOG(INFO)<<"======================Write the Reg 0xA4260014==============================";
    system("devmem 0xA4260010 32 0x01000000");
    
    LOG(INFO)<<"======================Write the Reg 0xA4260014==============================";
    system("devmem 0xA4260014 32 0x00000500");
    string xclbin = (iter % 2 == 0) ? xclbin1 : xclbin2;
    LOG(INFO) << "\n=== Iteration " << iter + 1 << ": " << xclbin << " ===";

    try {
      run_inference(xmodel, kernel, input_file, runner_num, count, xclbin);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Exception during inference: " << e.what();
    }
  }

  return 0;
}