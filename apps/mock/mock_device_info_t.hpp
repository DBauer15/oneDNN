#pragma once

#include <string>
#include <json/json.hpp>
#include "gpu/compute/device_info.hpp"

using namespace nlohmann;
using namespace dnnl::impl;

struct mock_device_info_t : gpu::compute::device_info_t {

    public:
    mock_device_info_t(
        std::string name = "Mock Compute Device",
        gpu::compute::gpu_arch_t gpu_arch = gpu::compute::gpu_arch_t::xe_hp,
        std::string runtime_version = "1.0.0",
        uint64_t extensions = 0,
        int32_t hw_threads_0 = 128,
        int32_t hw_threads_1 = 64,
        int32_t eu_count = 64,
        int32_t max_eus_per_wg = 8,
        int32_t max_subgroup_size = 256,
        size_t max_wg_size = 256,
        size_t llc_cache_size = 64
    ) : gpu::compute::device_info_t()
    {
        name_ = name; 
        gpu_arch_ = gpu_arch; 
        runtime_version_string = runtime_version;
        extensions_ = extensions;
        hw_threads_[0] = hw_threads_0;
        hw_threads_[1] = hw_threads_1;
        eu_count_ = eu_count;
        max_eus_per_wg_ = max_eus_per_wg;
        max_subgroup_size_ = max_subgroup_size;
        max_wg_size_ = max_wg_size;
        llc_cache_size_ = llc_cache_size;

        // Force ngen compatibility
        mayiuse_ngen_kernels_ = true;
        checked_ngen_kernels_ = true;
    };

    mock_device_info_t(json config) :
    mock_device_info_t(
        config["name"],
        gpu::compute::str2gpu_arch(std::string(config["gpu_arch"]).c_str()),
        config["runtime_version"],
        (uint64_t)config["extensions"],
        (int32_t)config["hw_threads"]["normal_grf"],
        (int32_t)config["hw_threads"]["large_grf"],
        (int32_t)config["eu_count"],
        (int32_t)config["max_eus_per_wg"],
        (int32_t)config["max_subgroup_size"],
        (size_t)config["max_wg_size"],
        (size_t)config["llc_cache_size"]
    ) {};

    protected:
    std::string runtime_version_string;

    virtual status_t init_device_name(engine_t *engine) override {
        return status_t::dnnl_success;
    };
    virtual status_t init_arch(engine_t *engine) override {
        return status_t::dnnl_success;
    };
    virtual status_t init_runtime_version(engine_t *engine) override {
        runtime_version_.set_from_string(runtime_version_string.c_str());
        return status_t::dnnl_success;
    };
    virtual status_t init_extensions(engine_t *engine) override {
        // TODO: - Do we need those?
        return status_t::dnnl_success;
    };
    virtual status_t init_attributes(engine_t *engine) override {
        return status_t::dnnl_success;
    };
};