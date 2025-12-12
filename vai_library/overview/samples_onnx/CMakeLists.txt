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


include_directories("${CMAKE_INSTALL_PREFIX}/include/onnxruntime")
include_directories("${CMAKE_SYSROOT}/usr/include/onnxruntime")
include_directories("${CMAKE_INSTALL_PREFIX}/include")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")
find_library(ORT_LIBRARY onnxruntime HINTS "${CMAKE_INSTALL_PREFIX}/lib" REQUIRED)

find_package(glog QUIET)
find_package(OpenCV COMPONENTS core highgui imgproc)
#find_package(vitis_ai_library )
find_package(vart)

if(NOT glog_FOUND)
  message(STATUS "cannot find glogConfig.cmake fallback to pkg-config")
  find_package(PkgConfig)
  pkg_search_module(PKG_GLOG REQUIRED IMPORTED_TARGET GLOBAL libglog)
  add_library(glog::glog ALIAS PkgConfig::PKG_GLOG)
endif(NOT glog_FOUND)

vai_overview_add_onnx(textmountain textmountain ) 
vai_overview_add_onnx(3DSegmentation 3DSegmentation ) 
vai_overview_add_onnx(reid  reid ) 
vai_overview_add_onnx(inception_v3_pt inception_v3_pt) 
vai_overview_add_onnx(resnet50_pt resnet50_pt) 

vai_overview_add_onnx(vehicle_type_resnet18_pt vehicle_type_resnet18_pt) 
vai_overview_add_onnx(segmentation segmentation REQUIRE xnnpp) 
vai_overview_add_onnx(movenet  movenet REQUIRE  json-c ) 
vai_overview_add_onnx(face_quality face_quality)
vai_overview_add_onnx(rcan rcan NO_ACC)

vai_overview_add_onnx(bevdet bevdet)
vai_overview_add_onnx(multitaskv3 multitaskv3   REQUIRE xnnpp) 
vai_overview_add_onnx(pointpillars pointpillars)
vai_overview_add_onnx(sesrs sesrs)
vai_overview_add_onnx(facemaskdet facemaskdet REQUIRE xnnpp)

vai_overview_add_onnx(unet2d unet2d)
vai_overview_add_onnx(yolovx_nano yolovx_nano)
vai_overview_add_onnx(yolov5 yolov5)
vai_overview_add_onnx(ofa_yolo ofa_yolo)
vai_overview_add_onnx(yolov7 yolov7)

vai_overview_add_onnx(yolov8 yolov8)

# vai_overview_add_onnx(inception_v3_multiep inception_v3_multiep)

install(DIRECTORY vehicle_classification_python DESTINATION ${SAMPLE_ONNX_INSTALL_PATH})
install(DIRECTORY resnet50_python DESTINATION ${SAMPLE_ONNX_INSTALL_PATH})
install(DIRECTORY squeezenet_python DESTINATION ${SAMPLE_ONNX_INSTALL_PATH})

