// single_thread_defog_xclbin_switch.cpp
#include <glog/logging.h>

#include <xir/attrs/attrs.hpp>
#include <xir/graph/graph.hpp>
#include "vart/runner_ext.hpp"
#include <thread>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <unistd.h>  // sleep

struct Job {
  std::string xclbin_path;
  std::string xmodel_path;
  std::string input_file;
  int execute_count = 200;
  std::string golden_file;
};

static xir::Subgraph* find_dpu_subgraph(xir::Graph* graph) {
  auto root = graph->get_root_subgraph();
  for (auto c : root->children_topological_sort()) {
    if (c->has_attr("device") && c->get_attr<std::string>("device") == "DPU") {
      return c;
    }
  }
  return nullptr;
}

static size_t total_outputs_bytes(const std::vector<vart::TensorBuffer*>& outputs) {
  size_t total = 0;
  for (auto* o : outputs) total += o->get_tensor()->get_data_size();
  return total;
}

static std::vector<uint8_t> read_file_exact(const std::string& path, size_t expected_size) {
  std::ifstream ifs(path, std::ios::binary);
  CHECK(ifs.good()) << "Failed to open file: " << path;

  std::vector<uint8_t> buf(expected_size);
  ifs.read(reinterpret_cast<char*>(buf.data()), expected_size);
  CHECK(ifs.gcount() == static_cast<std::streamsize>(expected_size))
      << "File size mismatch, file=" << path << " expected=" << expected_size
      << " got=" << ifs.gcount();
  return buf;
}

static std::pair<uint64_t, uint64_t> tb_linear_ptr(vart::TensorBuffer* tb) {
  return tb->data({0, 0, 0, 0});
}

static bool compare_outputs_with_golden_concat(const std::vector<vart::TensorBuffer*>& outputs,
                                              const std::vector<uint8_t>& golden,
                                              std::string* why) {
  size_t expected_total = total_outputs_bytes(outputs);
  CHECK_EQ(golden.size(), expected_total);

  size_t golden_off = 0;
  for (size_t oi = 0; oi < outputs.size(); ++oi) {
    auto* out_tb = outputs[oi];
    const size_t bytes = out_tb->get_tensor()->get_data_size();

    uint64_t ptr_u64 = 0, sz = 0;
    std::tie(ptr_u64, sz) = tb_linear_ptr(out_tb);
    CHECK(sz >= bytes);

    const uint8_t* out = reinterpret_cast<const uint8_t*>(ptr_u64);
    const uint8_t* ref = golden.data() + golden_off;

    for (size_t k = 0; k < bytes; ++k) {
      if (out[k] != ref[k]) {
        if (why) {
          *why = "mismatch: output[" + std::to_string(oi) + "] byte=" + std::to_string(k) +
                 " out=" + std::to_string(out[k]) + " ref=" + std::to_string(ref[k]) +
                 " golden_off=" + std::to_string(golden_off);
        }
        return false;
      }
    }
    golden_off += bytes;
  }
  return true;
}

static void fill_input_from_file(const std::vector<vart::TensorBuffer*>& inputs,
                                 const std::string& input_file) {
  CHECK(!inputs.empty());
  auto tb = inputs[0];

  const size_t batch = tb->get_tensor()->get_shape()[0];
  CHECK(batch > 0);

  const size_t total_size = tb->get_tensor()->get_data_size();
  const size_t size_per_batch = total_size / batch;

  // 只打开一次文件（比你原来在 batch 循环里反复 open 更合理）
  auto input_blob = read_file_exact(input_file, size_per_batch);

  for (size_t b = 0; b < batch; ++b) {
    uint64_t input_data = 0;
    uint64_t input_size = 0;
    std::tie(input_data, input_size) = tb->data({(int)b, 0, 0, 0});
    CHECK(input_size >= size_per_batch);
    std::memcpy(reinterpret_cast<void*>(input_data), input_blob.data(), size_per_batch);
  }
}

static void set_xclbin_firmware(const std::string& xclbin_path) {
  setenv("XLNX_VART_FIRMWARE", xclbin_path.c_str(), 1);
  LOG(INFO) << "Set XLNX_VART_FIRMWARE=" << xclbin_path;
}

// 单线程、单模型：每次 phase 都新建 runner，phase 结束 runner 自动释放
static void run_phase_single_model(const Job& job) {
  LOG(INFO) << "[phase] start"
            << " xclbin=" << job.xclbin_path
            << " xmodel=" << job.xmodel_path
            << " count=" << job.execute_count;

  // 0) 切 xclbin（必须在 create_runner 前）
  set_xclbin_firmware(job.xclbin_path);

  // 1) graph
  auto graph = xir::Graph::deserialize(job.xmodel_path);
  auto* dpu_subgraph = find_dpu_subgraph(graph.get());
  CHECK(dpu_subgraph != nullptr) << "No DPU subgraph found in model: " << job.xmodel_path;

  // 2) runner
  auto attrs = xir::Attrs::create();
  std::unique_ptr<vart::Runner> runner = vart::RunnerExt::create_runner(dpu_subgraph, attrs.get());
  auto* r = dynamic_cast<vart::RunnerExt*>(runner.get());
  CHECK(r != nullptr) << "Runner is not RunnerExt";

  auto inputs = r->get_inputs();
  auto outputs = r->get_outputs();
  CHECK_EQ(inputs.size(), 1u) << "Only support single input";

  // 3) golden 只读一次
  const size_t golden_bytes = total_outputs_bytes(outputs);
  auto golden_blob = read_file_exact(job.golden_file, golden_bytes);

  // 4) input
  fill_input_from_file(inputs, job.input_file);

  // 5) execute loop
  for (int c = 0; c < job.execute_count; ++c) {
    for (auto* in : inputs) {
      in->sync_for_write(0, in->get_tensor()->get_data_size());
    }

    // IMPORTANT: 这里如果你的 VART 支持 job id + wait，请务必接上 wait
    // auto jid = runner->execute_async(inputs, outputs);
    // runner->wait(jid.first /*or jid*/, -1);

    runner->execute_async(inputs, outputs);

    for (auto* out : outputs) {
      out->sync_for_read(0, out->get_tensor()->get_data_size());
    }

    std::string why;
    if (!compare_outputs_with_golden_concat(outputs, golden_blob, &why)) {
      LOG(ERROR) << "[phase] GOLDEN MISMATCH at iter=" << c << " : " << why;
      throw std::runtime_error("golden mismatch");
    }

    if ((c + 1) % 5 == 0) {
      LOG(INFO) << "[phase] finished " << (c + 1) << "/" << job.execute_count;
    }
  }

  LOG(INFO) << "[phase] done, runner will be released when leaving scope.";
}


// static void run_phase_in_new_thread(const Job& job) {
//   std::exception_ptr eptr = nullptr;

//   std::thread t([&] {
//     try {
//       run_phase_single_model(job);
//     } catch (...) {
//       eptr = std::current_exception();
//     }
//   });

//   // 這裡 join = “等這個 phase 的 thread 完全結束”
//   t.join();

//   if (eptr) std::rethrow_exception(eptr);
// }


int main(int argc, char* argv[]) {
  // google::InitGoogleLogging(argv[0]);

  const std::string xmodel_defog = "DeFog.xmodel";
  const std::string defog_input_file = "defog_input.bin";
  const std::string defog_golden_file = "defog_golden.bin";

  const std::string xclbin_dpu3d = "dpu3d.xclbin";
  const std::string xclbin_dpu4k = "dpu4k.xclbin";

  Job job;
  job.xmodel_path = xmodel_defog;
  job.input_file = defog_input_file;
  job.execute_count = 20;
  job.golden_file = defog_golden_file;

  for (int i = 0; i < 10000; ++i) {
    LOG(INFO) << "================ ITERATION=" << i << " ================";
    try {
      job.xclbin_path = xclbin_dpu3d;
      run_phase_single_model(job);   // phase thread #1 (create -> run -> join)

      sleep(1);

      job.xclbin_path = xclbin_dpu4k;
      run_phase_single_model(job);   // phase thread #2 (create -> run -> join)
    } catch (const std::exception& e) {
      LOG(ERROR) << "Fatal: " << e.what();
    }
  }
  return 0;
}