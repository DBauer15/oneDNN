#pragma once

#include <tuple>
#include <vector>
#include <memory>
#include "oneapi/dnnl/dnnl.hpp"
#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "gpu/compute/compute_engine.hpp"
#include "gpu/gpu_impl_list.hpp"
#include "gpu/jit/conv/gen_convolution.hpp"

#include "mock_device_info_t.hpp"


using namespace dnnl::impl;

struct mock_engine : public gpu::compute::compute_engine_t {


    mock_engine() : gpu::compute::compute_engine_t(engine_kind::gpu, runtime_kind::ocl, 0) { 
        init();
    };

    virtual device_id_t device_id() const override {
        return std::tuple(0,0,0);
    };

    virtual engine_id_t engine_id() const override {
        return 0;
    };

    virtual status_t create_memory_storage(
        memory_storage_t **storage, 
        unsigned flags, 
        size_t size,
        void *handle) override {
        
        return status::success;
    };

    virtual status_t create_stream(stream_t **stream, unsigned flags) override {
        return status::success;
    ;}

    virtual const impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        static const impl_list_item_t empty_list[] = {nullptr};

        assert(!"unknown primitive kind");
        return empty_list;
    };

    virtual const impl_list_item_t *
    get_concat_implementation_list() const override {
        static const impl_list_item_t empty_list[] = {nullptr};

        assert(!"unknown primitive kind");
        return empty_list;
    };

    virtual const impl_list_item_t *
    get_sum_implementation_list() const override {
        static const impl_list_item_t empty_list[] = {nullptr};

        assert(!"unknown primitive kind");
        return empty_list;
    };

    virtual const impl_list_item_t *get_implementation_list(
            const op_desc_t *desc) const override {

        static const impl_list_item_t empty_list[] = {nullptr};

        if (desc->kind == primitive_kind::convolution) {
            static const impl_list_item_t conv_list[] = {
                INSTANCE(gpu::jit::gen_convolution_fwd_t)
                nullptr
                };
            return conv_list;
        } else {
            return empty_list;
        }
    };

    virtual bool mayiuse_f16_accumulator_with_f16() const override { 
        return false; 
    }

    /* Compute Engine Function */
    virtual status_t create_kernel(gpu::compute::kernel_t *kernel, gpu::jit::jit_generator_base *jitter, cache_blob_t cache_blob) const override {
        return status::unimplemented;
    };

    virtual status_t create_kernels(std::vector<gpu::compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const gpu::compute::kernel_ctx_t &kernel_ctx,
            cache_blob_t cache_blob) const override {
        
        return status::unimplemented;
    };

    virtual std::function<void(void *)> get_program_list_deleter() const override {
        return [](void *p) {};
    };

    protected:
    virtual status_t init_device_info() override {
        device_info_ = std::make_shared<mock_device_info_t>();
        device_info_->init(this);
        return status::success;
    };
    
};