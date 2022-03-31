/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_CONFIG_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_CONFIG_HPP

#include <memory>
#include <string>
#include <vector>
#include "graph.hpp"
#include "util/general_object.hpp"

namespace sc {
// todo(zhichen): replaced by any map
struct graph_config {
    std::vector<reflection::shared_general_object_t> op_cfgs_;
    // maybe anther config item in the future
};
namespace tuner {
struct config_space;
} // namespace tuner

namespace graph {
SC_INTERNAL_API std::unique_ptr<tuner::config_space>
extract_tuning_space_from_graph(context_ptr ctx, const sc_graph_t &g,
        std::string space_name = "op_configs");

} // namespace graph
} // namespace sc

#endif