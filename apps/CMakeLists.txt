#===============================================================================
# Copyright 2016-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_BUILD_TYPE Debug)

append(CMAKE_CXX_FLAGS "-fno-operator-names")

include_directories_with_host_compiler(${PROJECT_SOURCE_DIR}/src)
include_directories_with_host_compiler(${PROJECT_SOURCE_DIR}/include)
include_directories_with_host_compiler(${PROJECT_SOURCE_DIR}/apps/extern)

file(GLOB_RECURSE sources *.cpp *.c)
file(GLOB_RECURSE headers *.hpp *.h)

# Place build results directly in ./build directory
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

add_executable(ir_dump ${sources})
target_link_libraries(ir_dump ${DNNL_LIBRARY_NAME})

# Copy the config file
# TODO: Not working yet - fix eventually
# file(GLOB CONFIG_FILES "${CMAKE_SOURCE_DIR}/apps/config/*.json")
# add_custom_command(TARGET ir_dump
#                 POST_BUILD
#                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                 ${CONFIG_FILES}
#                 $<TARGET_FILE_DIR:ir_dump>)