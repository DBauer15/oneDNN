#include <iostream>
#include <numeric>
#include <stdexcept>

#include <CL/cl.h>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/gen_convolution.cpp"
#include "gpu/jit/conv/gen_convolution.hpp"
#include "gpu/jit/conv/hw_config.hpp"
#include "gpu/jit/conv/ir.cpp"
#include "gpu/jit/conv/kernel_builder.hpp"
#include "gpu/jit/conv/kernel_info.hpp"

#include "mock/mock_engine.hpp"
#include "mock/mock_device_info_t.hpp"

using namespace std;
using namespace dnnl;
using namespace dnnl::impl::gpu::jit;

using tag = memory::format_tag;
using dt = memory::data_type;

void make_conv() {

        // Created mocked-up engine
        //mock_device_info_t device_info;
        mock_engine mengine;
        dnnl::engine engine(&mengine);
        //dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        

        // Tensor dimensions.
        const memory::dim N = 3, // batch size
                IC = 32, // input channels
                IH = 13, // input height
                IW = 13, // input width
                OC = 64, // output channels
                KH = 3, // weights height
                KW = 3, // weights width
                PH_L = 1, // height padding: left
                PH_R = 1, // height padding: right
                PW_L = 1, // width padding: left
                PW_R = 1, // width padding: right
                SH = 4, // height-wise stride
                SW = 4, // width-wise stride
                OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
                OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width

        // Source (src), weights, bias, and destination (dst) tensors
        // dimensions.
        memory::dims src_dims = {N, IC, IH, IW};
        memory::dims weights_dims = {OC, IC, KH, KW};
        memory::dims bias_dims = {OC};
        memory::dims dst_dims = {N, OC, OH, OW};

        // Strides, padding dimensions.
        memory::dims strides_dims = {SH, SW};
        memory::dims padding_dims_l = {PH_L, PW_L};
        memory::dims padding_dims_r = {PH_R, PW_R};

        // Allocate buffers.
        // std::vector<float> src_data(product(src_dims));
        // std::vector<float> weights_data(product(weights_dims));
        // std::vector<float> bias_data(OC);
        // std::vector<float> dst_data(product(dst_dims));

        // Initialize src, weights, and dst tensors.
        // std::generate(src_data.begin(), src_data.end(), []() {
        //         static int i = 0;
        //         return std::cos(i++ / 10.f);
        // });
        // std::generate(weights_data.begin(), weights_data.end(), []() {
        //         static int i = 0;
        //         return std::sin(i++ * 2.f);
        // });
        // std::generate(bias_data.begin(), bias_data.end(), []() {
        //         static int i = 0;
        //         return std::tanh(float(i++));
        // });

        // Create memory objects for tensor data (src, weights, dst). In this
        // example, NCHW layout is assumed for src and dst, and OIHW for weights.
        // auto user_src_mem = memory({src_dims, dt::f32, tag::nchw}, engine);
        // auto user_weights_mem = memory({weights_dims, dt::f32, tag::oihw}, engine);
        // auto user_dst_mem = memory({dst_dims, dt::f32, tag::nchw}, engine);

        // Create memory descriptors with format_tag::any for the primitive. This
        // enables the convolution primitive to choose memory layouts for an
        // optimized primitive implementation, and these layouts may differ from the
        // ones provided by the user.
        auto conv_src_md = memory::desc(src_dims, dt::f32, tag::any);
        auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::any);
        auto conv_dst_md = memory::desc(dst_dims, dt::f32, tag::any);

        // Create memory descriptor and memory object for input bias.
        auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);
        //auto user_bias_mem = memory(user_bias_md, engine);

        // Write data to memory object's handle.
        // write_to_dnnl_memory(src_data.data(), user_src_mem);
        // write_to_dnnl_memory(weights_data.data(), user_weights_mem);
        // write_to_dnnl_memory(bias_data.data(), user_bias_mem);

        // Create operation descriptor.
        auto conv_desc = convolution_forward::desc(dnnl::prop_kind::forward_training,
                algorithm::convolution_direct, conv_src_md, conv_weights_md,
                user_bias_md, conv_dst_md, strides_dims, padding_dims_l,
                padding_dims_r);
        

        // Create primitive post-ops (ReLU).
        // const float scale = 1.f;
        // const float alpha = 0.f;
        // const float beta = 0.f;
        // post_ops conv_ops;
        // conv_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
        primitive_attr conv_attr;
        // conv_attr.set_post_ops(conv_ops);

        // Create primitive descriptor.
        auto conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, engine);
        
}

int main(int argc, char **argv) {
//     jit_generator<gpu_gen_t::XeHP> gen;
//     ir_printer_t printer(std::cout);

//     // Create a conv config
//     mock_conv_config_t config;
//     config.init_with_raw_values(); //TODO: Implement this

//     // Create a kernel info
//     kernel_info_t info;
//     info.set_id(kernel_id_t::convolution);

    // Create a kernel builder from the mocked-up information
    //kernel_builder_t builder(config, pd, info);

    // Dump the kernel to std::cout
    //printer.visit(builder.stmt());

    make_conv();
}