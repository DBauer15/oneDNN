#include <iostream>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <CL/cl.h>
#include <json/json.hpp>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/gen_convolution.cpp"
#include "gpu/jit/conv/gen_convolution.hpp"
#include "gpu/jit/conv/hw_config.hpp"
#include "gpu/jit/conv/ir.cpp"
#include "gpu/jit/conv/kernel_builder.hpp"
#include "gpu/jit/conv/kernel_info.hpp"

#include "mock/mock_engine_t.hpp"
#include "mock/mock_device_info_t.hpp"

using namespace nlohmann;
using namespace dnnl;
using namespace dnnl::impl::gpu::jit;

using tag = memory::format_tag;
using dt = memory::data_type;

primitive_desc make_conv(json config, engine engine) {        

        // Tensor dimensions.
        const memory::dim N = config["N"], // batch size
                IC = config["IC"], // input channels
                IH = config["IH"], // input height
                IW = config["IW"], // input width
                OC = config["OC"], // output channels
                KH = config["KH"], // weights height
                KW = config["KW"], // weights width
                PH_L = config["PH_L"], // height padding: left
                PH_R = config["PH_R"], // height padding: right
                PW_L = config["PW_L"], // width padding: left
                PW_R = config["PW_R"], // width padding: right
                SH = config["SH"], // height-wise stride
                SW = config["SW"], // width-wise stride
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

        // Create memory descriptors with format_tag::any for the primitive. This
        // enables the convolution primitive to choose memory layouts for an
        // optimized primitive implementation, and these layouts may differ from the
        // ones provided by the user.
        auto conv_src_md = memory::desc(src_dims, dt::f32, tag::any);
        auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::any);
        auto conv_dst_md = memory::desc(dst_dims, dt::f32, tag::any);

        // Create memory descriptor and memory object for input bias.
        auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);


        // Create operation descriptor.
        auto conv_desc = convolution_forward::desc(dnnl::prop_kind::forward_training,
                algorithm::convolution_direct, 
                conv_src_md, 
                conv_weights_md,
                //user_bias_md, //TODO: using bias results in segfault - investigate
                conv_dst_md, 
                strides_dims, 
                padding_dims_l,
                padding_dims_r);        

        // Create primitive post-ops
        primitive_attr conv_attr;

        // Create primitive descriptor.
        auto conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, engine);

        return conv_pd;
}

void dump_ir(primitive_desc pd, engine e) {

        std::shared_ptr<primitive_t> p;

        // This will trigger JIT compilation
        pd.get()->impl()->create_primitive(p, e.get());
}

json load_config(std::string filename) {
        json config;
        std::ifstream config_file(filename);
        if (!config_file.is_open() || !config_file.good())
                throw std::runtime_error("could not open config file");

        config_file >> config;
        if (config_file.is_open())
                config_file.close();

        return config;
}

void print_usage() {
        std::cout << "---------------------------" << std::endl;
        std::cout << "oneDNN -- IR DUMP" << std::endl;
        std::cout << "---------------------------\n" << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << "./ir_dump [convolution-config] [hardware-config]" << std::endl;
}


int main(int argc, char **argv) {
        // Read config
        if (argc < 3) { print_usage(); throw std::runtime_error("no config files provided"); }
        
        json config_conv = load_config(argv[1]);
        json config_hardware = load_config(argv[2]);

        // Created mocked-up engine
        mock_engine_t *mengine = new mock_engine_t(config_hardware["hardware"]);
        engine engine(mengine);

        // Create primitive description for single convolution
        primitive_desc pd = make_conv(config_conv["convolution"], engine);

        // Dump the IR code
        dump_ir(pd, engine);
}