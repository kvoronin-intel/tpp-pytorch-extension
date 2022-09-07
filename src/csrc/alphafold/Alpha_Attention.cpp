
#include<immintrin.h>
#include<iostream>
#include<stdio.h>
#include<torch/extension.h>
#include<tuple>
#include<omp.h>
#include <cmath>
#include<libxsmm.h>
#include<libxsmm_intrinsics_x86.h>

#include <ATen/record_function.h>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace pcl;
#include "tensor_helper.h"

#define QKV_BLOCKSIZE 128
#define A_BLOCKSIZE 128
#define C_BLOCKSIZE 128

REGISTER_SCOPE(alpha_q_gemm, "alpha_q_gemm");
REGISTER_SCOPE(alpha_k_gemm, "alpha_k_gemm");
REGISTER_SCOPE(alpha_v_gemm, "alpha_v_gemm");

REGISTER_SCOPE(alpha_a_gemm, "alpha_a_gemm");
REGISTER_SCOPE(alpha_c_gemm, "alpha_c_gemm");

at::Tensor fused_gating_attention_fwd(at::Tensor& q_data, at::Tensor& m_data, at::Tensor& bias, at::Tensor& nonbatched_bias,
                                    at::Tensor& query_w, at::Tensor& key_w, at::Tensor& value_w, 
                                    at::Tensor& gating_w, at::Tensor& gating_b, at::Tensor& output_w, at::Tensor& output_b, int key_dim, int value_dim){
    
    GlobalPass _gp(FWD);
    if (q_data[0].dtype() == at::kFloat) {
        typedef float T;
        #include "fused_gating_attention_fwd_tmpl.h"
    } else {
        typedef bfloat16 T;
        // #include "Conv1dOpti_forward_libxsmm_tmpl_bf16.h"
    }
}


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
REGISTER_SUBMODULE(_alpha_attention, m){
m.def("forward", &fused_gating_attention_fwd, "Gating attention forward");
}