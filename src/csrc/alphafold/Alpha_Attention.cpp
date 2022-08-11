
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

at::Tensor fused_gating_attention_fwd(at::Tensor& q_data, at::Tensor& m_data, at::Tensor& bias, at::Tensor& nonbatched_bias,
                                    at::Tensor& query_w, at::Tensor& key_w, at::Tensor& value_w, 
                                    at::Tensor& gating_w, at::Tensor& gating_b, at::Tensor& output_w, at::Tensor& output_b, int key_dim, int value_dim){
    
    int64_t B_t = q_data.size(0);                                       /* Batch (512) */
    int64_t S_t = q_data.size(1);                                       /* Query (764) */
    int64_t HS_t = q_data.size(2);                                      /* Channels (256) */

    int64_t N_t = query_w.size(1);                                      /* number of heads (8) */
    int64_t H_t = query_w.size(2);                                      /* head size (32) */

    // auto output = q_data.new_empty({B_t, S_t, HS_t});                   /* [512, 764, 256] */

    // auto q = q_data.new_empty({B_t, S_t, N_t, H_t});                    /* [512, 764, 8, 32] */
    // auto k = q_data.new_empty({B_t, S_t, N_t, H_t});                    /* [512, 764, 8, 32] */
    // float* k_a = k.data_ptr<float, 4>();
    // float* m_data_a = m_data.data_ptr<float, 3>();

    // auto v = q_data.new_empty({B_t, S_t, N_t, H_t});                    /* [512, 764, 8, 32] */

    // auto logits = q_data.new_empty({B_t, N_t, S_t, S_t});               /* [512, 8, 764, 764] */
    // auto weights = q_data.new_empty({B_t, N_t, S_t, S_t});              /* [512, 8, 764, 764] */
    // auto weighted_avg = q_data.new_empty({B_t, S_t, N_t, H_t});         /* [512, 764, 8, 32] */

    // auto gate_values = q_data.new_empty({B_t, S_t, N_t, value_dim});    /* [512, 764, 8, 32] */
    
    auto q = at::mul(at::einsum("bqa,ahc->bqhc", {q_data, query_w}), (1.0/sqrt(key_dim))) ;     /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8, 32] */

    auto k = at::einsum("bka,ahc->bkhc", {m_data, key_w});                                      /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8, 32] */
    // #pragma omp parallel for collapse(2)
    // for(size_t i=0; i < B_t; i++){
    //     for(size_t j=0; j < S_t; j++){
        // [1, 8, 32] = [1, 256] * [256, 8, 32] k =32 , br = 256/32, n =32
    //         k.index({i, j, at::indexing::slice(0, N_t), at::indexing::slice(0, H_t)}) = at::mm(m_data.index({i, j, at::indexing::slice(0, HS_t)}), key_w);
    //     }
    // }

    auto v = at::einsum("bka,ahc->bkhc", {m_data, value_w});                                    /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8, 32] */

    auto logits = at::add(at::einsum("bqhc,bkhc->bhqk", {q, k}), bias);                         /* [512, 8, 764, 764]  = [512, 764, 8, 32] * [512, 764, 8, 32] + [512, 1, 1, 764] */

    if (nonbatched_bias.size(0) > 0)
        logits = at::add(logits, at::unsqueeze(nonbatched_bias, 0));                       /* [512, 8, 764, 764]  = [512, 8, 764, 764] + [1, 8, 764, 764] */
        
    auto weights = at::_softmax(logits, -1, false);                                             /* [512, 8, 764, 764] = [512, 8, 764, 764] */

    auto weighted_avg = at::einsum("bhqk,bkhc->bqhc", {weights, v});                            /* [512, 764, 8, 32]  = [512, 8, 764, 764] * [512, 764, 8, 32] */

    auto gate_values = at::sigmoid(at::add(at::einsum("bqc,chv->bqhv", {q_data, gating_w}), gating_b));      /* [512, 764, 8, 32]  = [512, 764, 32] * [32, 8, 764] + [8, 32]*/

    weighted_avg = at::mul(weighted_avg, gate_values);                                     /* [512, 764, 8, 32]  = [512, 764, 8, 32] * [512, 764, 8, 32] */

    auto output = at::add(at::einsum("bqhc,hco->bqo", {weighted_avg, output_w}), output_b);     /* [512, 764, 256]  = [512, 764, 8, 32] * [8, 32, 256] + [256] */

    return output;

    // int64_t b_t = q_data.size(0);                    /* Batch (512) */
    // int64_t q_t = q_data.size(1);                    /* Query (764) */
    // int64_t k_t = m_data.size(1);                    /* Key (764) */
    // int64_t a_t = q_data.size(2);                  /* Channels (256) */

    // int64_t h_t = query_w.size(1);                  /* number of heads (8) */
    // int64_t c_t = query_w.size(2);                  /* head channels (32) */

    // auto output = q_data.new_empty({b_t,q_t,a_t});

    // auto q = q_data.new_empty({b_t,q_t,h_t,c_t});
    // auto k = q_data.new_empty({b_t,k_t,h_t,c_t});
    // auto v = q_data.new_empty({b_t,k_t,h_t,c_t});

    // auto logits = q_data.new_empty({b_t,h_t,q_t,k_t});
    // auto weights = q_data.new_empty({b_t,h_t,q_t,k_t});
    // auto weighted_avg = q_data.new_empty({b_t,q_t,h_t,c_t});

    // auto gate_values = q_data.new_empty({b_t,q_t,h_t,value_dim});

    // q = at::mul(at::einsum("bqa,ahc->bqhc", {q_data, query_w}), (1.0/sqrt(key_dim))) ;
    // k = at::einsum("bka,ahc->bkhc", {m_data, key_w});
    // v = at::einsum("bka,ahc->bkhc", {m_data, value_w});

    // logits = at::add(at::einsum("bqhc,bkhc->bhqk", {q, k}), bias);

    // if (nonbatched_bias.size(0) > 0)
    //     logits = at::add(logits, at::unsqueeze(nonbatched_bias, 0));
    
    // weights = at::_softmax(logits, -1, false);

    // weighted_avg = at::einsum("bhqk,bkhc->bqhc", {weights, v});

    // gate_values = at::sigmoid(at::add(at::einsum("bqc,chv->bqhv", {q_data, gating_w}), gating_b));

    // weighted_avg = at::mul(weighted_avg, gate_values);

    // output = at::add(at::einsum("bqhc,hco->bqo", {weighted_avg, output_w}), output_b);

    // return output;
}



// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
REGISTER_SUBMODULE(_alpha_attention, m){
m.def("forward", &fused_gating_attention_fwd, "Gating attention forward");
}