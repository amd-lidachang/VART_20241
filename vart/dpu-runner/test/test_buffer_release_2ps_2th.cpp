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

#include <atomic>
#include <sys/wait.h>
#include <signal.h>

static size_t total_outputs_bytes(const std::vector<vart::TensorBuffer*>& outputs) {
  size_t total = 0;
  for (auto* o : outputs) total += o->get_tensor()->get_data_size();
  return total;
}

static void set_xclbin_firmware(const std::string& xclbin_path) {
  setenv("XLNX_VART_FIRMWARE", xclbin_path.c_str(), 1);
  LOG(INFO) << "Set XLNX_VART_FIRMWARE=" << xclbin_path;
}

static xir::Subgraph* find_dpu_subgraph(xir::Graph* graph) {
  auto root = graph->get_root_subgraph();
  for (auto c : root->children_topological_sort()) {
    if (c->has_attr("device") && c->get_attr<std::string>("device") == "DPU") {
      return c;
    }
  }
  return nullptr;
}

static std::pair<uint64_t, uint64_t> tb_linear_ptr(vart::TensorBuffer* tb) {
  return tb->data({0, 0, 0, 0});
}

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
          {
      std::cerr << *why << "\nPress ENTER to continue..." << std::flush;
      std::cin.clear();
      std::string line;
      std::getline(std::cin, line);
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



static int run_model_in_current_thread(
    const std::string& tag,                 // 方便打日志：defog / hls
    const std::string& xmodel_path,
    const std::string& input_file,
    const std::string& golden_file,
    int execute_count) {

  auto graph = xir::Graph::deserialize(xmodel_path);
  auto* dpu_subgraph = find_dpu_subgraph(graph.get());
  CHECK(dpu_subgraph != nullptr) << "No DPU subgraph found in model: " << xmodel_path;

  auto attrs = xir::Attrs::create();
  std::unique_ptr<vart::Runner> runner = vart::RunnerExt::create_runner(dpu_subgraph, attrs.get());
  auto* r = dynamic_cast<vart::RunnerExt*>(runner.get());
  CHECK(r != nullptr) << "Runner is not RunnerExt";

  auto inputs = r->get_inputs();
  auto outputs = r->get_outputs();
  CHECK_EQ(inputs.size(), 1u);

  const size_t golden_bytes = total_outputs_bytes(outputs);
  auto golden_blob = read_file_exact(golden_file, golden_bytes);

  // 每次执行前填一次输入（你也可以挪到循环内，看需求）
  fill_input_from_file(inputs, input_file);

  for (int c = 0; c < execute_count; ++c) {
    for (auto* in : inputs) {
      in->sync_for_write(0, in->get_tensor()->get_data_size());
    }

    // 强烈建议 wait（你现在代码没 wait，很多情况下会出问题）
    auto jid = r->execute_async(inputs, outputs);
    r->wait(jid.first, -1);

    for (auto* out : outputs) {
      out->sync_for_read(0, out->get_tensor()->get_data_size());
    }

    std::string why;
    bool ok = compare_outputs_with_golden_concat(outputs, golden_blob, &why);
    if (!ok) {
      LOG(ERROR) << "[" << tag << "] GOLDEN MISMATCH iter=" << c << " : " << why;
      return 2;
    }
  }

  LOG(INFO) << "[" << tag << "] done";
  return 0;
}


static int run_two_models_in_current_process(
    const std::string& xclbin_path,

    const std::string& defog_xmodel,
    const std::string& defog_input,
    const std::string& defog_golden,

    const std::string& hls_xmodel,
    const std::string& hls_input,
    const std::string& hls_golden,

    int execute_count) {

  set_xclbin_firmware(xclbin_path);

  std::atomic<int> rc_defog{0};
  std::atomic<int> rc_hls{0};

  std::thread th_defog([&] {
    try {
      rc_defog = run_model_in_current_thread("defog", defog_xmodel, defog_input, defog_golden, execute_count);
    } catch (const std::exception& e) {
      LOG(ERROR) << "[defog] exception: " << e.what();
      rc_defog = 3;
    } catch (...) {
      LOG(ERROR) << "[defog] unknown exception";
      rc_defog = 4;
    }
  });

  std::thread th_hls([&] {
    try {
      rc_hls = run_model_in_current_thread("hls", hls_xmodel, hls_input, hls_golden, execute_count);
    } catch (const std::exception& e) {
      LOG(ERROR) << "[hls] exception: " << e.what();
      rc_hls = 3;
    } catch (...) {
      LOG(ERROR) << "[hls] unknown exception";
      rc_hls = 4;
    }
  });

  th_defog.join();
  th_hls.join();

  if (rc_defog != 0) return rc_defog.load();
  if (rc_hls != 0) return rc_hls.load();
  return 0;
}


// ===== 你原本的工具函數：find_dpu_subgraph / total_outputs_bytes / read_file_exact
// / tb_linear_ptr / compare_outputs_with_golden_concat / fill_input_from_file
// 這些都可以原封不動保留 =====




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


static int run_one_phase_in_child_process_and_wait(
    const std::string& xclbin_path,

    const std::string& defog_xmodel,
    const std::string& defog_input,
    const std::string& defog_golden,

    const std::string& hls_xmodel,
    const std::string& hls_input,
    const std::string& hls_golden,

    int execute_count) {
  pid_t pid = fork();
  CHECK(pid >= 0) << "fork() failed: " << std::strerror(errno);

  if (pid == 0) {
    // 重要：fork 后在子进程里初始化 glog（尤其你父进程也用 glog 时）
    google::InitGoogleLogging("child");

    int rc = 0;
    try {
      rc = run_two_models_in_current_process(
          xclbin_path,
          defog_xmodel, defog_input, defog_golden,
          hls_xmodel,   hls_input,   hls_golden,
          execute_count);
    } catch (const std::exception& e) {
      LOG(ERROR) << "[child] exception: " << e.what();
      rc = 5;
    } catch (...) {
      LOG(ERROR) << "[child] unknown exception";
      rc = 6;
    }
    _exit(rc);
  }

  int status = 0;
  CHECK(waitpid(pid, &status, 0) == pid) << "waitpid() failed: " << std::strerror(errno);

  if (WIFEXITED(status)) return WEXITSTATUS(status);
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return 99;
}
// -------------------- Worker: 不再持有 runner/graph/attrs --------------------
/*class Worker {
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
*/





int main(int argc, char* argv[]) {


  const std::string defog_input_file = "defog_input.bin";
  const std::string defog_xmodel     = "DeFog.xmodel";
  const std::string defog_golden     = "defog_golden.bin";

  const std::string hls_input_file   = "hls_input.bin";
  const std::string hls_xmodel       = "HLS.xmodel";
  const std::string hls_golden       = "hls_golden.bin";

  const std::string xclbin_dpu3d     = "dpu3d.xclbin";
  const std::string xclbin_dpu4k     = "dpu4k.xclbin";

  for (int i = 0; i < 10000; ++i) {
    LOG(INFO) << "================ ITERATION=" << i << " ================";

    int rc1 = run_one_phase_in_child_process_and_wait(
        xclbin_dpu3d,
        defog_xmodel, defog_input_file, defog_golden,
        hls_xmodel,   hls_input_file,   hls_golden,
        1);
    if (rc1 != 0) break;

    int rc2 = run_one_phase_in_child_process_and_wait(
        xclbin_dpu4k,
        defog_xmodel, defog_input_file, defog_golden,
        hls_xmodel,   hls_input_file,   hls_golden,
        1);
    if (rc2 != 0) break;
  }
  return 0;
}