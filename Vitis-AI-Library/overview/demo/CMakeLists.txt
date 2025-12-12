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


#tensorflow example
install(FILES Custom_OP_Demo/tensorflow2_example/op_Mylayer/Makefile
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/tensorflow2_example/op_Mylayer/)
install(FILES Custom_OP_Demo/tensorflow2_example/op_Mylayer/my_Mylayer_op.cpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/tensorflow2_example/op_Mylayer/)
install(FILES Custom_OP_Demo/tensorflow2_example/tf2_custom_op_graph_runner/build.sh
        PERMISSIONS
          OWNER_READ
          OWNER_WRITE
          OWNER_EXECUTE
          GROUP_READ
          GROUP_EXECUTE
          WORLD_READ
          WORLD_EXECUTE
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/tensorflow2_example/tf2_custom_op_graph_runner/)
install(FILES Custom_OP_Demo/tensorflow2_example/tf2_custom_op_graph_runner/readme
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/tensorflow2_example/tf2_custom_op_graph_runner/)
install(FILES Custom_OP_Demo/tensorflow2_example/tf2_custom_op_graph_runner/tf2_custom_op_graph_runner.cpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/tensorflow2_example/tf2_custom_op_graph_runner/)
install(FILES Custom_OP_Demo/tensorflow2_example/run.sh
        PERMISSIONS
          OWNER_READ
          OWNER_WRITE
          OWNER_EXECUTE
          GROUP_READ
          GROUP_EXECUTE
          WORLD_READ
          WORLD_EXECUTE
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/tensorflow2_example/)

#pytorch example
install(FILES Custom_OP_Demo/pytorch_example/op_registration/op_PPScatter/CMakeLists.txt
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/op_registration/op_PPScatter/)
install(FILES Custom_OP_Demo/pytorch_example/op_registration/op_PPScatter/Makefile
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/op_registration/op_PPScatter/)
install(FILES Custom_OP_Demo/pytorch_example/op_registration/op_PPScatter/my_PPScatter_op.cpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/op_registration/op_PPScatter/)
install(FILES Custom_OP_Demo/pytorch_example/op_registration/op_registration.sh
        PERMISSIONS
          OWNER_READ
          OWNER_WRITE
          OWNER_EXECUTE
          GROUP_READ
          GROUP_EXECUTE
          WORLD_READ
          WORLD_EXECUTE
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/op_registration/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/build.sh
        PERMISSIONS
          OWNER_READ
          OWNER_WRITE
          OWNER_EXECUTE
          GROUP_READ
          GROUP_EXECUTE
          WORLD_READ
          WORLD_EXECUTE
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/readme
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/data/sample_pointpillars.bin
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/data/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/anchor.cpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/anchor.hpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/helper.cpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/helper.hpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/main.cpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/parse_display_result.cpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/pointpillars.hpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/pointpillars_post.cpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/pointpillars_post.hpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/preprocess.cpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/)
install(FILES Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/preprocess.hpp
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/)
install(FILES Custom_OP_Demo/pytorch_example/run.sh
        PERMISSIONS
          OWNER_READ
          OWNER_WRITE
          OWNER_EXECUTE
          GROUP_READ
          GROUP_EXECUTE
          WORLD_READ
          WORLD_EXECUTE
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/)

#build tensorflow example
add_library(Mylayer SHARED Custom_OP_Demo/tensorflow2_example/op_Mylayer/my_Mylayer_op.cpp)
target_link_libraries(Mylayer PUBLIC glog::glog vart::runner xir::xir ${PROJECT_NAME}::runner_helper ${PROJECT_NAME}::cpu_task)
set_target_properties(Mylayer PROPERTIES LIBRARY_OUTPUT_NAME vart_op_imp_Mylayer)
install(TARGETS Mylayer
        PERMISSIONS
          OWNER_READ
          OWNER_WRITE
          OWNER_EXECUTE
          GROUP_READ
          GROUP_EXECUTE
          WORLD_READ
          WORLD_EXECUTE
        LIBRARY DESTINATION lib/)

find_package(OpenCV COMPONENTS opencv_core opencv_video opencv_videoio
                               opencv_imgproc opencv_imgcodecs opencv_highgui)
if(NOT OpenCV_FOUND)
  find_package(
    OpenCV_LIBS
    opencv_core
    opencv_video
    opencv_videoio
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui)
endif(NOT OpenCV_FOUND)
find_package(PkgConfig REQUIRED)

add_executable(tf2_custom_op_graph_runner Custom_OP_Demo/tensorflow2_example/tf2_custom_op_graph_runner/tf2_custom_op_graph_runner.cpp)
target_link_libraries(tf2_custom_op_graph_runner glog::glog vart::runner xir::xir ${PROJECT_NAME}::runner_helper ${PROJECT_NAME}::graph_runner opencv_core opencv_imgcodecs opencv_imgproc)
install(TARGETS tf2_custom_op_graph_runner
        PERMISSIONS
          OWNER_READ
          OWNER_WRITE
          OWNER_EXECUTE
          GROUP_READ
          GROUP_EXECUTE
          WORLD_READ
          WORLD_EXECUTE
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/tensorflow2_example/tf2_custom_op_graph_runner/)

#build pytorch example
add_library(PPScatter SHARED Custom_OP_Demo/pytorch_example/op_registration/op_PPScatter/my_PPScatter_op.cpp)
target_link_libraries(PPScatter PUBLIC glog::glog vart::runner xir::xir ${PROJECT_NAME}::runner_helper ${PROJECT_NAME}::cpu_task)
set_target_properties(PPScatter PROPERTIES LIBRARY_OUTPUT_NAME vart_op_imp_PPScatterV2)
install(TARGETS PPScatter
        PERMISSIONS
          OWNER_READ
          OWNER_WRITE
          OWNER_EXECUTE
          GROUP_READ
          GROUP_EXECUTE
          WORLD_READ
          WORLD_EXECUTE
        LIBRARY DESTINATION lib/)

set(sample_name "sample_pointpillars_graph_runner")
set(CMAKE_CXX_STANDARD 17)
find_package(xir)
find_package(Eigen3 REQUIRED)
find_package(OpenCV )
find_package(glog)

if(CPACK_DEBIAN_PACKAGE_ARCHITECTURE STREQUAL "arm64")
  add_compile_options(-Wno-error=attributes -Wno-strict-aliasing)
endif()

add_compile_options( -Wno-array-bounds )

add_executable(${sample_name}
   Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/anchor.cpp
   Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/helper.cpp
   Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/preprocess.cpp
   Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/pointpillars_post.cpp
   Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/parse_display_result.cpp
   Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/src/main.cpp
)
target_include_directories(${sample_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${sample_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${sample_name} PUBLIC ${OpenCV_LIBS}
                      glog::glog Eigen3::Eigen ${PROJECT_NAME}::graph_runner vart::runner xir::xir pthread)
install(TARGETS ${sample_name}
        PERMISSIONS
          OWNER_READ
          OWNER_WRITE
          OWNER_EXECUTE
          GROUP_READ
          GROUP_EXECUTE
          WORLD_READ
          WORLD_EXECUTE
        DESTINATION share/vitis_ai_library/demo/Custom_OP_Demo/pytorch_example/pointpillars_graph_runner/build/)
