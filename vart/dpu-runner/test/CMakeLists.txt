#
# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
option(ENABLE_PACK_TRD "Enable create tar.gz for TRD" OFF)
aux_source_directory(. TEST_SRC)

execute_process(
  COMMAND
    bash -ex ${CMAKE_CURRENT_SOURCE_DIR}/convert_word_list.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/word_list.txt
    ${CMAKE_CURRENT_BINARY_DIR}/word_list.inc)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

find_package(Eigen3)
find_package(OpenCV REQUIRED)
if(MSVC)
  set(TEST_SRCS test_dpu_runner.cpp test_dpu_runner_mt.cpp)
else(MSVC)
  # for WINDOWS, because word_list.inc is not generated, we remove resnet50.cpp
  set(TEST_SRCS test_dpu_runner.cpp resnet50.cpp test_dpu_runner_mt.cpp)
endif(MSVC)
foreach(FNAME ${TEST_SRCS})
  get_filename_component(F_PREFIX ${FNAME} NAME_WE)
  set(ELF ${F_PREFIX})

  add_executable(${ELF} ${FNAME})
  target_link_libraries(
    ${ELF}
    ${COMPONENT_NAME}
    xir::xir
    glog::glog
    util
    mem-manager
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT})
endforeach()

if(MSVC)

else(MSVC)
  # TRD packaging is not supported for windows yet.
  if(ENABLE_PACK_TRD)
    install(
      CODE "execute_process (
    COMMAND env CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR} bash ${CMAKE_CURRENT_SOURCE_DIR}/pack_for_trd.sh
)")
  endif(ENABLE_PACK_TRD)
endif(MSVC)

if(XRT_FOUND AND NOT MSVC)
  Add_executable(xvdpu_profiler ${CMAKE_CURRENT_SOURCE_DIR}/xvdpu_profiler.cpp)
  target_link_libraries(
    xvdpu_profiler
    xir::xir
    glog::glog
    XRT::xrt_core
    XRT::xrt_coreutil
    xrt-device-handle
    buffer-object
    runner
    dpu-runner)
  install(TARGETS xvdpu_profiler DESTINATION bin)
endif(XRT_FOUND AND NOT MSVC)
# add_executable (resnet50_zero_copy ../samples/resnet50_zero_copy/resnet50.cpp)
# target_link_libraries (resnet50_zero_copy ${COMPONENT_NAME}
# ${PROJECT_NAME}::runner dpu-controller opencv_core opencv_video opencv_videoio
# opencv_imgproc opencv_imgcodecs opencv_highgui xrt_coreutil )

include_directories(
  ${CMAKE_SYSROOT}/usr/include
  ${CMAKE_SYSROOT}/usr/include/xrt
  ${CMAKE_SYSROOT}/usr/include/xrt/xrt
  ${CMAKE_SYSROOT}/usr/include/python3.10 
)
add_executable(test_dpu_runner_mindray test_dpu_runner_mindray.cpp)
add_executable(test_dpu_runner_th test_dpu_runner.cpp)
add_executable(test_buffer_100 test_buffer.cpp)
add_executable(test_buffer_release test_buffer_release.cpp)
add_executable(test_buffer_sysreset test_buffer_sysreset.cpp)
add_executable(test_buffer_double test_buffer_double.cpp)
add_executable(test_buffer_double_gaochong test_buffer_double_gaochong.cpp)
add_executable(test_buffer_read_reg test_buffer_read_reg.cpp)
add_library(dpu_mr MODULE mr_runner.cpp)

target_link_libraries(test_dpu_runner_mindray ${COMPONENT_NAME}
${PROJECT_NAME}::runner dpu-controller opencv_core opencv_video opencv_videoio
opencv_imgproc opencv_imgcodecs opencv_highgui xrt_coreutil)
target_link_libraries(test_dpu_runner_th ${COMPONENT_NAME}
${PROJECT_NAME}::runner dpu-controller opencv_core opencv_video opencv_videoio
opencv_imgproc opencv_imgcodecs opencv_highgui xrt_coreutil)
target_link_libraries(test_buffer_100 ${COMPONENT_NAME}
${PROJECT_NAME}::runner dpu-controller opencv_core opencv_video opencv_videoio
opencv_imgproc opencv_imgcodecs opencv_highgui xrt_coreutil vart-xrt-device-handle)
target_link_libraries(test_buffer_double ${COMPONENT_NAME}
${PROJECT_NAME}::runner dpu-controller opencv_core opencv_video opencv_videoio
opencv_imgproc opencv_imgcodecs opencv_highgui xrt_coreutil vart-xrt-device-handle)
target_link_libraries(test_buffer_release ${COMPONENT_NAME}
${PROJECT_NAME}::runner dpu-controller opencv_core opencv_video opencv_videoio
opencv_imgproc opencv_imgcodecs opencv_highgui xrt_coreutil vart-xrt-device-handle )

target_link_libraries(test_buffer_sysreset ${COMPONENT_NAME}
${PROJECT_NAME}::runner dpu-controller opencv_core opencv_video opencv_videoio
opencv_imgproc opencv_imgcodecs opencv_highgui xrt_coreutil vart-xrt-device-handle uuid)

target_link_libraries(test_buffer_double_gaochong ${COMPONENT_NAME}
${PROJECT_NAME}::runner dpu-controller opencv_core opencv_video opencv_videoio
opencv_imgproc opencv_imgcodecs opencv_highgui xrt_coreutil vart-xrt-device-handle runner-assistant)

target_link_libraries(test_buffer_read_reg ${COMPONENT_NAME}
${PROJECT_NAME}::runner dpu-controller opencv_core opencv_video opencv_videoio
opencv_imgproc opencv_imgcodecs opencv_highgui xrt_coreutil vart-xrt-device-handle runner-assistant)

target_link_libraries(dpu_mr ${COMPONENT_NAME}
${PROJECT_NAME}::runner dpu-controller opencv_core opencv_video opencv_videoio
opencv_imgproc opencv_imgcodecs opencv_highgui xrt_coreutil vart-xrt-device-handle runner-assistant python3.10)

set_target_properties(dpu_mr PROPERTIES PREFIX "")
