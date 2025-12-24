#include <glog/logging.h>

#include <xir/graph/graph.hpp>
#include <xir/attrs/attrs.hpp>
#include "vart/runner_ext.hpp"

#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <unistd.h>
#include <fcntl.h>

#include <chrono>
#include <sstream>
    // open, O_RDONLY, O_SYNC
#include <sys/mman.h>   // mmap, munmap, PROT_READ, MAP_SHARED, MAP_FAILED
   // close, sysconf, _SC_PAGESIZE
#include <cerrno>       // errno
#include <cstring>      // std::strerror
#include <cstdint>      // uint64_t
// // 读整个文件到 string
// static std::string slurp_file(const std::string& path) {
//   std::ifstream ifs(path);
//   if (!ifs.good()) return {};
//   std::ostringstream oss;
//   oss << ifs.rdbuf();
//   return oss.str();
// }

// 从 kds_custat_raw 判断是否存在 busy (status == busy_code)
// static bool kds_any_cu_busy(uint32_t busy_code,
//                            std::string* debug_line /*optional*/) {
//   const std::string path = "/sys/class/drm/card0/device/kds_custat_raw";
//   auto txt = slurp_file(path);
//   if (txt.empty()) {
//     // 读不到就当作“无法判断”，由上层决定要不要放行
//     if (debug_line) *debug_line = "cannot read " + path;
//     return false;
//   }

//   std::istringstream iss(txt);
//   std::string line;
//   while (std::getline(iss, line)) {
//     if (line.empty()) continue;

//     // 以逗号分割，取第5列 status
//     // 格式: a,b,name,addr,status,usage
//     std::vector<std::string> cols;
//     cols.reserve(6);
//     size_t start = 0;
//     while (true) {
//       size_t pos = line.find(',', start);
//       if (pos == std::string::npos) {
//         cols.emplace_back(line.substr(start));
//         break;
//       }
//       cols.emplace_back(line.substr(start, pos - start));
//       start = pos + 1;
//     }

//     if (cols.size() < 5) continue;

//     // cols,[object Object], 应该是 "0x1" / "0x4" 这种
//     uint32_t status = 0;
//     try {
//       status = static_cast<uint32_t>(std::stoul(cols[4], nullptr, 0));
//     } catch (...) {
//       continue;
//     }

//     if (status == busy_code) {
//       if (debug_line) *debug_line = line;
//       return true;
//     }
//   }
//   return false;
// }

// 等到所有 CU 不处于 busy_code；timeout_ms<0 表示永不超时
// static void wait_kds_cu_not_busy(uint32_t busy_code = 0x1,
//                                 int64_t timeout_ms = -1,
//                                 int poll_interval_ms = 20) {
//   using clock = std::chrono::steady_clock;
//   auto t0 = clock::now();

//   while (true) {
//     std::string why;
//     bool busy = kds_any_cu_busy(busy_code, &why);

//     if (!busy) return;

//     if (timeout_ms >= 0) {
//       auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
//                          clock::now() - t0)
//                          .count();
//       if (elapsed > timeout_ms) {
//         throw std::runtime_error("Timeout waiting KDS CU status != 0x1, last busy line: " + why);
//       }
//     }

//     // 需要的话可降低打印频率
//     LOG(INFO) << "[main] KDS CU busy (status==0x" << std::hex << busy_code
//               << std::dec << "), wait... line=" << why;

//     std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
//   }
// }

static xir::Subgraph* find_dpu_subgraph(xir::Graph* graph) {
  auto root = graph->get_root_subgraph();
  for (auto c : root->children_topological_sort()) {
    if (c->has_attr("device") && c->get_attr<std::string>("device") == "DPU") {
      return c;
    }
  }
  return nullptr;
}

/*static uint64_t read_devmem_u64(uint64_t phys_addr) {
  const uint64_t page_size = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
  const uint64_t page_base = phys_addr & ~(page_size - 1);
  const uint64_t page_off  = phys_addr - page_base;

  int fd = open("/dev/mem", O_RDONLY | O_SYNC);
  CHECK(fd >= 0) << "open(/dev/mem) failed: " << std::strerror(errno);

  void* map = mmap(nullptr, page_size, PROT_READ, MAP_SHARED, fd, page_base);
  CHECK(map != MAP_FAILED) << "mmap failed: " << std::strerror(errno);

  volatile uint64_t* p = reinterpret_cast<volatile uint64_t*>(
      reinterpret_cast<uint8_t*>(map) + page_off);

  uint64_t v = *p;

  munmap(map, page_size);
  close(fd);
  return v;
}*/


static void set_xclbin_firmware(const std::string& xclbin_path) {
  setenv("XLNX_VART_FIRMWARE", xclbin_path.c_str(), 1);
  LOG(INFO) << "Set XLNX_VART_FIRMWARE=" << xclbin_path;
}


static std::vector<uint8_t> read_file_exact(const std::string& path, size_t expected_size) {
  std::ifstream ifs(path, std::ios::binary);
  CHECK(ifs.good()) << "Failed to open golden file: " << path;

  std::vector<uint8_t> buf(expected_size);
  ifs.read(reinterpret_cast<char*>(buf.data()), expected_size);
  CHECK(ifs.gcount() == static_cast<std::streamsize>(expected_size))
      << "Golden size mismatch, file=" << path
      << " expected=" << expected_size
      << " got=" << ifs.gcount();
  return buf;
}
static size_t total_outputs_bytes(const std::vector<vart::TensorBuffer*>& outputs) {
  size_t total = 0;
  for (auto* o : outputs) total += o->get_tensor()->get_data_size();
  return total;
}


struct RunnerBundle {
  std::unique_ptr<xir::Graph> graph;
  std::unique_ptr<xir::Attrs> attrs;
  std::unique_ptr<vart::Runner> runner;   // owning (主線程持有)
  vart::RunnerExt* r = nullptr;           // cached pointer

  std::vector<vart::TensorBuffer*> inputs;
  std::vector<vart::TensorBuffer*> outputs;
  uint64_t extra_devmem_value = 0;
  uint64_t extra_devmem_addr = 0;
  std::shared_ptr<std::vector<uint8_t>> golden;
};

static RunnerBundle make_runner_bundle(const std::string& xmodel_path,
                                      const std::string& golden_file, const size_t device_core_id) {
  RunnerBundle b;

  b.graph = xir::Graph::deserialize(xmodel_path);
  auto* dpu_subgraph = find_dpu_subgraph(b.graph.get());
  CHECK(dpu_subgraph != nullptr) << "No DPU subgraph found in model: " << xmodel_path;

  b.attrs = xir::Attrs::create();
  b.attrs->set_attr<size_t>("__device_core_id__", device_core_id);
  b.attrs->set_attr<size_t>("__device_id__", 0);
 
  b.runner = vart::RunnerExt::create_runner(dpu_subgraph, b.attrs.get());
 
  b.r = dynamic_cast<vart::RunnerExt*>(b.runner.get());
  CHECK(b.r != nullptr) << "Runner is not RunnerExt";

   /*// --- ADD: choose devmem address by xmodel name ---
  uint64_t devmem_addr = 0;
  if (xmodel_path.find("DeFog.xmodel") != std::string::npos) {
    devmem_addr = 0x50100043040;
  } else if (xmodel_path.find("HLS.xmodel") != std::string::npos) {
    devmem_addr = 0x50102d37040;
  }

  if (devmem_addr != 0) {
    uint64_t val = read_devmem_u64(devmem_addr);
    // 可选：保存到 bundle（需要你在 RunnerBundle 里加字段）
    b.extra_devmem_addr = devmem_addr;
    b.extra_devmem_value = val;

    LOG(INFO) << "devmem[0x" << std::hex << devmem_addr << "] = 0x" << val << std::dec;
  } else {
    LOG(INFO) << "xmodel_path does not match DeFog.xmodel/HLS.xmodel, skip devmem read: "
              << xmodel_path;
  }
  // --- END ADD ---*/

  b.inputs = b.r->get_inputs();
  b.outputs = b.r->get_outputs();
  CHECK_EQ(b.inputs.size(), 1u) << "Only support single input";

  const size_t golden_bytes = total_outputs_bytes(b.outputs);
  b.golden = std::make_shared<std::vector<uint8_t>>(
      read_file_exact(golden_file, golden_bytes));

  return b;
}
// -------------------- Task: worker 只執行，不創建/銷毀 runner --------------------
struct Task {
  vart::Runner* runner = nullptr;  // 非 owning，生命週期由主線程管理
  std::vector<vart::TensorBuffer*> inputs;
  std::vector<vart::TensorBuffer*> outputs;

  std::string input_file;
  int execute_count = 0;

  // golden 已經在主線程讀好，worker 只拿來比對
  std::shared_ptr<std::vector<uint8_t>> golden;
};

class Worker {
 public:
  explicit Worker(std::string name) : name_(std::move(name)) {
    th_ = std::thread([this] { this->loop(); });
  }

  ~Worker() {
    stop();
    if (th_.joinable()) th_.join();
  }

  void submit(Task t) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_idle_.wait(lk, [&]{ return state_ == State::IDLE; });
    task_ = std::move(t);
    state_ = State::HAS_TASK;
    cv_.notify_one();
  }

  void wait_done() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&]{ return state_ == State::IDLE; });
    if (!last_error_.empty()) {
      throw std::runtime_error(name_ + " failed: " + last_error_);
    }
  }

  void stop() {
    std::unique_lock<std::mutex> lk(mu_);
    if (stop_) return;
    stop_ = true;
    cv_.notify_one();
  }

 private:
  enum class State { IDLE, HAS_TASK, RUNNING };

  void loop() {
    for (;;) {
      Task t;
      {
        std::unique_lock<std::mutex> lk(mu_);
        state_ = State::IDLE;
        last_error_.clear();
        cv_idle_.notify_all();
        cv_done_.notify_all();

        cv_.wait(lk, [&]{ return stop_ || state_ == State::HAS_TASK; });
        if (stop_) return;

        t = std::move(task_);
        state_ = State::RUNNING;
      }

      try {
        run_task(t);
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(mu_);
        last_error_ = e.what();
      } catch (...) {
        std::lock_guard<std::mutex> lk(mu_);
        last_error_ = "unknown error";
      }
    }
  }







static std::pair<uint64_t, uint64_t> tb_linear_ptr(vart::TensorBuffer* tb) {
  return tb->data({0, 0, 0, 0});
}



// -------------------- Worker: 不再持有 runner/graph/attrs --------------------



 static bool compare_outputs_with_golden_concat(
    const std::vector<vart::TensorBuffer*>& outputs,
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
                 " out=" + std::to_string(out[k]) +
                 " ref=" + std::to_string(ref[k]) +
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
  auto tensor = tb->get_tensor();
  auto shape = tensor->get_shape();
  CHECK(!shape.empty());
  const size_t batch = inputs[0]->get_tensor()->get_shape()[0];
  // const size_t total_size = tensor->get_data_size();
  CHECK(batch > 0);
  // const size_t size_per_batch = total_size / batch;
  size_t size_per_batch = inputs[0]->get_tensor()->get_data_size() / batch;

  for (size_t b = 0; b < batch; ++b) {
    uint64_t input_data = 0;
    uint64_t input_size = 0;
    std::tie(input_data, input_size) = tb->data({(int)b, 0, 0, 0});
    CHECK(input_size >= size_per_batch);

    std::ifstream ifs(input_file, std::ios::binary);
    CHECK(ifs.good()) << "Failed to open input file: " << input_file;
    CHECK(ifs.read(reinterpret_cast<char*>(input_data), size_per_batch).good())
        << "Failed to read input data, file=" << input_file;
  }
}
  void run_task(const Task& t) {
    CHECK(t.runner != nullptr);
    CHECK(!t.inputs.empty());
    CHECK(t.golden && !t.golden->empty());

    LOG(INFO) << "[" << name_ << "] start task, count=" << t.execute_count;

    // input 填充也可以放 worker，但你說希望 runner 銷毀在主線程，
    // 這裡是否放 worker 不影響；我保留在這裡，保持你原本邏輯。
    fill_input_from_file(t.inputs, t.input_file);

    // sleep(0.5);
    for (int c = 0; c < t.execute_count; ++c) {
      for (auto* in : t.inputs) {
        in->sync_for_write(0, in->get_tensor()->get_data_size());
      }

      // 建議你接 wait（依你 VART 版本調整）
      // auto jid = t.runner->execute_async(t.inputs, t.outputs);
      // t.runner->wait(jid.first /*or jid*/, -1);
      t.runner->execute_async(t.inputs, t.outputs);

      for (auto* out : t.outputs) {
        out->sync_for_read(0, out->get_tensor()->get_data_size());
      }

      std::string why;
      bool ok = compare_outputs_with_golden_concat(t.outputs, *t.golden, &why);
      if (!ok) {
        LOG(ERROR) << "[" << name_ << "] GOLDEN MISMATCH iter=" << c << " : " << why;
        throw std::runtime_error("golden mismatch");
      }

      if ((c + 1) % 5 == 0) {
        LOG(INFO) << "[" << name_ << "] finished " << (c + 1) << "/" << t.execute_count;
      }
    }

    LOG(INFO) << "[" << name_ << "] task done.";
  }

 private:
  std::string name_;
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable cv_idle_;
  std::condition_variable cv_done_;

  State state_{State::IDLE};
  Task task_;
  bool stop_{false};
  std::string last_error_;
};


static void run_phase_two_models(
    Worker& w_defog,
    Worker& w_hls,
    const std::string& xclbin_path,

    const std::string& xmodel_defog,
    const std::string& defog_input,
    const std::string& defog_golden,
    int defog_count,

    const std::string& xmodel_hls,
    const std::string& hls_input,
    const std::string& hls_golden,
    int hls_count) {

  // (1) 主線程切 xclbin：一定要在 create_runner 前
  // wait_kds_cu_not_busy(/*busy_code=*/0x1, /*timeout_ms=*/200, /*poll_interval_ms=*/20);

  set_xclbin_firmware(xclbin_path);

  // (2) 主線程建立兩個 runner（同一個 xclbin 之下）
  auto defog = make_runner_bundle(xmodel_defog, defog_golden, 0);
  auto hls   = make_runner_bundle(xmodel_hls,   hls_golden, 1);

  // (3) 組 task：worker 只用 runner 指標 + buffers + golden 做 execute/比對
  Task t0;
  t0.runner = defog.runner.get();
  t0.inputs = defog.inputs;
  t0.outputs = defog.outputs;
  t0.input_file = defog_input;
  t0.execute_count = defog_count;
  t0.golden = defog.golden;

  Task t1;
  t1.runner = hls.runner.get();
  t1.inputs = hls.inputs;
  t1.outputs = hls.outputs;
  t1.input_file = hls_input;
  t1.execute_count = hls_count;
  t1.golden = hls.golden;

  // (4) 並行跑（兩個 worker）
  w_defog.submit(std::move(t0));
  sleep(0.5);
  w_hls.submit(std::move(t1));

  // (5) 主線程等待兩邊結束（確保 worker 不再使用 runner）
  w_defog.wait_done();
  w_hls.wait_done();

  // (6) 你要的關鍵：主線程「在切換 xclbin 的中間」銷毀 runner/graph/attrs
  LOG(INFO) << "[main] destroying runners in main thread before switching xclbin...";

  // sleep(1);
  hls.runner.reset();
  hls.attrs.reset();
  hls.graph.reset();
  sleep(1);    
  defog.runner.reset();
  defog.attrs.reset();
  defog.graph.release();
  sleep(0.5);
 
  LOG(INFO) << "[main] phase done for xclbin=" << xclbin_path;
}
// ===== 你原本的工具函數：find_dpu_subgraph / total_outputs_bytes / read_file_exact
// / tb_linear_ptr / compare_outputs_with_golden_concat / fill_input_from_file
// 這些都可以原封不動保留 =====

int main(int argc, char* argv[]) {


  const std::string defog_input_file = "defog_input.bin";
  const std::string hls_input_file   = "hls_input.bin";

  const std::string xmodel_defog = "DeFog.xmodel";
  const std::string xmodel_hls   = "HLS.xmodel";

  const std::string xclbin_dpu3d = "dpu3d.xclbin";
  const std::string xclbin_dpu4k = "dpu4k.xclbin";

  Worker w_defog("worker-0(DeFog)");
  Worker w_hls("worker-1(HLS)");

  for (int i = 0; i < 10000; ++i) {
    LOG(INFO) << "================ ITERATION=" << i << " ================";
    try {
      run_phase_two_models(w_defog, w_hls,
          xclbin_dpu3d,
          xmodel_defog, defog_input_file, "defog_golden.bin", 1,
          xmodel_hls,   hls_input_file,   "hls_golden.bin",   1);
      
      run_phase_two_models(w_defog, w_hls,
          xclbin_dpu4k,
          xmodel_defog, defog_input_file, "defog_golden.bin", 1,
          xmodel_hls,   hls_input_file,   "hls_golden.bin",   1);

    } catch (const std::exception& e) {
      LOG(ERROR) << "Fatal: " << e.what();
    }
  }
  return 0;
}