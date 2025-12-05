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
get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

aux_source_directory(src SRC_FILES)
aux_source_directory(src/op SRC_FILES)
aux_source_directory(src/alg SRC_FILES)
aux_source_directory(src/vis SRC_FILES)
aux_source_directory(src/api SRC_FILES)
aux_source_directory(src/sample SRC_FILES)

add_library(${COMPONENT_NAME}_without_symbol ${SRC_FILES})

if(BUILD_SHARED_LIBS)
  target_compile_definitions(${COMPONENT_NAME}_without_symbol
                             PUBLIC -DVART_CPU_RUNNER_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(${COMPONENT_NAME}_without_symbol
                             PUBLIC -DVART_CPU_RUNNER_USE_DLL=0)
endif(BUILD_SHARED_LIBS)


target_include_directories(
  ${COMPONENT_NAME}_without_symbol
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
          ${CMAKE_CURRENT_SOURCE_DIR}/src/alg
          ${CMAKE_CURRENT_SOURCE_DIR}/src/op
          ${CMAKE_CURRENT_SOURCE_DIR}/src/vis
          ${CMAKE_CURRENT_SOURCE_DIR}/src/api
          ${CMAKE_CURRENT_SOURCE_DIR}/src/sample)
target_link_libraries(
  ${COMPONENT_NAME}_without_symbol
  PUBLIC util runner mem-manager xir::xir unilog::unilog)


set(dump_symbol_cmd)
if(MSVC)
    set(dump_symbol_cmd "dumpbin" "/SYMBOLS")
else()
    set(dump_symbol_cmd "objdump" "-t")
endif()
add_custom_command(
  TARGET ${COMPONENT_NAME}_without_symbol
  POST_BUILD
  COMMAND ${dump_symbol_cmd} $<TARGET_FILE:${COMPONENT_NAME}_without_symbol> > cpu_runner_symbol.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
  TARGET ${COMPONENT_NAME}_without_symbol
  POST_BUILD
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/src/symbol/symbol.py ${CMAKE_CURRENT_BINARY_DIR}/symbols.cpp cpu_runner_symbol.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/symbols.cpp
)

add_library(${COMPONENT_NAME} ${CMAKE_CURRENT_BINARY_DIR}/symbols.cpp)
target_link_libraries(${COMPONENT_NAME} ${COMPONENT_NAME}_without_symbol)

set_target_properties(
  ${COMPONENT_NAME}
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME})

install(
  TARGETS ${COMPONENT_NAME}_without_symbol
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT cpu
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT cpu
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(
  FILES include/cpu_std_inc.hpp
        include/cpu_base_inc.hpp
        include/cpu_util.hpp
        include/load_cfg.hpp
        include/cpu_cfg.hpp
        include/cpu_types.hpp
        include/cpu_dimension.hpp
        include/cpu_helper.hpp
        include/cpu_reg_func.hpp
        include/cpu_op_base.hpp
        include/cpu_tensor_buffer.hpp
        include/cpu_op_visitor.hpp
        include/cpu_runner.hpp
  COMPONENT cpu
  DESTINATION include/vart/cpu)

install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT cpu
  DESTINATION share/cmake/${PROJECT_NAME})

if(BUILD_TEST)
  add_subdirectory(test)
endif()
