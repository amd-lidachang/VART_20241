#include <glog/logging.h> 
#include <google/protobuf/message.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <cstdlib>

#include <xir/tensor/tensor.hpp>
#include "vart/dpu/vitis_dpu_runner_factory.hpp"
#include "vart/runner_ext.hpp"
#include "xir/xrt_device_handle.hpp"
# include <xrt/xrt_bo.h>
# include <xrt/xrt_device.h>
# include <xrt/xrt_kernel.h>
#include "vart/assistant/xrt_bo_tensor_buffer.hpp"
#include "vart/zero_copy_helper.hpp"
#include "xir/graph/graph.hpp"

int main(int argc, char* argv[])
{
    if (argc < 4) {
        LOG(ERROR) << "[3d] Model path not provided.";
        return -1;
    }
    std::string model_path = argv[1];
    std::string tmp = argv[2];
    std::string tmp_1 = argv[3];

    std::ifstream infile_0(tmp, std::ios::binary);
    std::ifstream infile_1(tmp_1, std::ios::binary);
    CHECK(infile_0.good()) << "Failed to open " << tmp;
    CHECK(infile_1.good()) << "Failed to open " << tmp_1;

    auto ret = setenv("XLNX_VART_FIRMWARE", "/usr/lib/rp.xclbin", 1);
    LOG(INFO) << "[3d] setenv XLNX_VART_FIRMWARE ret=" << ret;
    auto curVer = getenv("XLNX_VART_FIRMWARE");
    LOG(INFO) << "[3d] XLNX_VART_FIRMWARE=" << curVer;
    auto m_graph3d = xir::Graph::deserialize(model_path);
    auto root3d = m_graph3d->get_root_subgraph();
    xir::Subgraph* s3d = nullptr;
    for (auto c : root3d->get_children()) {
        if (c->get_attr<std::string>("device") == "DPU") {
            s3d = c;
            break;
        }
    }
    auto m_attrs3d = xir::Attrs::create();
    m_attrs3d->set_attr<size_t>("__device_core_id__", 0);
    m_attrs3d->set_attr<size_t>("__device_id__", 0);
    auto m_runner3d = vart::RunnerExt::create_runner(s3d, m_attrs3d.get());
    std::vector<const xir::Tensor*> inputTensors3d = m_runner3d->get_input_tensors();
    std::vector<const xir::Tensor*> outputTensors3d = m_runner3d->get_output_tensors();
 
    // 输入是2维 输出是1维
    // batch==1
    if (inputTensors3d.size() != 2 || outputTensors3d.size() != 1
        || inputTensors3d[0]->get_shape().size() != 4 || outputTensors3d[0]->get_shape().size() != 4
        || inputTensors3d[0]->get_shape()[0] != 1 || outputTensors3d[0]->get_shape()[0] != 1) {
        LOG(ERROR) << "[3d] inputTensors size=" << inputTensors3d.size()
            << " outputTensors size=" << outputTensors3d.size();
        return -1;
    }

    auto m_device3d = std::make_shared<xrt::device>(0);
    auto input_tensor_buffer_size3d = inputTensors3d[0]->get_data_size() / inputTensors3d[0]->get_shape()[0];
    auto m_inputBo3d = std::make_unique<xrt::bo>(*m_device3d, input_tensor_buffer_size3d*2, 1);
    if (m_inputBo3d == nullptr || m_inputBo3d->size() == 0) {
        LOG(ERROR) << "[3d] Failed to allocate XRT buffer object." << std::endl;
        return -1;
    }
    auto m_inputBuffer3d = vart::assistant::XrtBoTensorBuffer::create(
        vart::xrt_bo_t{ m_device3d.get(), m_inputBo3d.get() }, inputTensors3d[0]);
    // auto m_inputPhyAddr3d = m_inputBuffer3d->data_phy({ 0, 0, 0, 0 }).first;
    auto m_inputVirtAddr3d = m_inputBuffer3d->data({ 0, 0, 0, 0 }).first;
    LOG(INFO) << "[3d] m_inputBuffer=" << m_inputBuffer3d->to_string()
        // << " m_inputPhyAddr=" << std::hex << "0x" << m_inputPhyAddr3d << std::dec
        << " m_inputVirtAddr=" << std::hex << "0x" << m_inputVirtAddr3d << std::dec
        << " request bo_size=" << input_tensor_buffer_size3d
        << " allocated bo_size= " << m_inputBo3d->size()
        << " input tensor size=" << inputTensors3d[0]->get_data_size();

    auto m_inputBo3d_2 = std::make_unique<xrt::bo>(*m_device3d, input_tensor_buffer_size3d, 1);
    if (m_inputBo3d_2 == nullptr || m_inputBo3d_2->size() == 0) {
        LOG(ERROR) << "[3d] Failed to allocate XRT buffer object." << std::endl;
        return -1;
    }


    //*****input use dachang method */
    std::vector<char> buffer0(input_tensor_buffer_size3d);

    std::vector<char> buffer1(input_tensor_buffer_size3d);
        CHECK(infile_0.read(buffer0.data(), input_tensor_buffer_size3d).good())
        << "Failed to read input data from " << tmp;
    CHECK(infile_1.read(buffer1.data(), input_tensor_buffer_size3d).good())
        << "Failed to read input data from " << tmp_1;
      auto input = m_runner3d->get_inputs();
      uint64_t input_data0 = 0u, input_size0 = 0u;
      uint64_t input_data1 = 0u, input_size1 = 0u;
      std::tie(input_data0, input_size0) = input[0]->data({0, 0, 0, 0});
      std::tie(input_data1, input_size1) = input[1]->data({0, 0, 0, 0});

      std::memcpy(reinterpret_cast<void*>(input_data0), buffer0.data(), input_size0);
      std::memcpy(reinterpret_cast<void*>(input_data1), buffer1.data(), input_size1);


      for (auto in : input) {
      in->sync_for_write(0, in->get_tensor()->get_data_size());
    }

    /////////////
    auto m_inputBuffer3d_2 = vart::assistant::XrtBoTensorBuffer::create(
        vart::xrt_bo_t{ m_device3d.get(), m_inputBo3d_2.get() }, inputTensors3d[1]);
    // 7. 获取物理地址和该 TensorBuffer 实际对应的数据大小（batch index = 0）
    // auto m_inputPhyAddr3d_2 = m_inputBuffer3d_2->data_phy({ 0, 0, 0, 0 }).first;
    auto m_inputVirtAddr3d_2 = m_inputBuffer3d_2->data({ 0, 0, 0, 0 }).first;
    LOG(INFO) << "[3d] m_inputBuffer_2=" << m_inputBuffer3d_2->to_string()
        // << " m_inputPhyAddr_2=" << std::hex << "0x" << m_inputPhyAddr3d_2 << std::dec
        << " m_inputVirtAddr_2=" << std::hex << "0x" << m_inputVirtAddr3d_2 << std::dec
        << " request bo_size=" << input_tensor_buffer_size3d
        << " allocated bo_size= " << m_inputBo3d_2->size()
        << " input tensor size=" << inputTensors3d[0]->get_data_size();

    // ****output use xrt::bo method*****
    auto output_tensor_buffer_size3d = outputTensors3d[0]->get_data_size() / outputTensors3d[0]->get_shape()[0];
    auto m_outputBo3d = std::make_unique<xrt::bo>(*m_device3d, output_tensor_buffer_size3d, 1);
    if (m_outputBo3d == nullptr || m_outputBo3d->size() == 0) {
        LOG(ERROR) << "[3d] Failed to allocate XRT output buffer object." << std::endl;
        return -1;
    }
    LOG(INFO) << "[3d] output tensor=" << outputTensors3d[0]->to_string();
    auto m_outputBuffer3d = vart::assistant::XrtBoTensorBuffer::create(
        vart::xrt_bo_t{ m_device3d.get(), m_outputBo3d.get() }, outputTensors3d[0]);
    auto m_outputPhyAddr3d = m_outputBuffer3d->data_phy({ 0, 0, 0, 0 }).first;
    auto m_outputVirtAddr3d = m_outputBo3d->map();
    LOG(INFO) << "[3d] m_outputBuffer=" << m_outputBuffer3d->to_string()
        << " m_outputPhyAddr=" << std::hex << "0x" << m_outputPhyAddr3d << std::dec
        << " m_outputVirtAddr=" << std::hex << "0x" << m_outputVirtAddr3d << std::dec
        << " request bo_size=" << output_tensor_buffer_size3d
        << " allocated bo_size= " << m_outputBo3d->size();

   auto ret3d = m_runner3d->execute_async(input,  { m_outputBuffer3d.get() });
    auto pt3d = m_runner3d->wait(ret3d.first, -1);
    LOG(INFO) << "[3d] execute_async ret=" << ret3d.second << " jobid=" << ret3d.first << " pt3d=" << pt3d;
    m_outputBuffer3d->sync_for_read(0, output_tensor_buffer_size3d);
    uint64_t output_data;
    std::size_t output_size;
    std::tie(output_data, output_size) = m_outputBuffer3d->data({0, 0, 0, 0});

        std::ofstream outfile("output.bin", std::ios::binary | std::ios::trunc);
    if (!outfile.is_open())
    {
        LOG(ERROR) << "[3d] outfile=" << tmp << " open file failed";
        return -1;
    }
    auto virtAddr = m_outputVirtAddr3d;
    auto outSize = output_tensor_buffer_size3d;
    ret = outfile.write(reinterpret_cast<const char*>(virtAddr), outSize).good();

    LOG(INFO) << "[3d] DumpInterFrame success=" << ret
        << "\n input size=" << input_tensor_buffer_size3d
        << "\n input size2=" << input_tensor_buffer_size3d
        << "\n input virt addr=" << std::hex << "0x" << m_inputVirtAddr3d << std::dec
        << "\n input virt addr2=" << std::hex << "0x" << m_inputVirtAddr3d_2 << std::dec
        << "\n output virt addr=" << std::hex << "0x" << m_outputVirtAddr3d << std::dec
        << "\n output phy addr=" << std::hex << "0x" << m_outputPhyAddr3d << std::dec
        << "\n output size=" << m_outputVirtAddr3d
        << "\n output_size =" << output_size
        << "\n errno=" << errno
        << "\n errmsg=" << strerror(errno);
        return 0;


}