
RECORD_FUNCTION("Gating attention forward", std::vector<c10::IValue>({q_data, m_data}));    // For recording time
    
    int64_t B_t = q_data.size(0);                                       /* Batch (512) */
    int64_t S_t = q_data.size(1);                                       /* Query (764) */
    int64_t HS_t = q_data.size(2);                                      /* Channels (256) */

    int64_t N_t = query_w.size(1);                                      /* number of heads (8) */
    int64_t H_t = query_w.size(2);                                      /* head size (32) */

    auto q = q_data.new_empty({B_t, S_t, N_t, H_t});                    /* [512, 764, 8, 32] */
    DECL_VLA_PTR_PT(T, q_a, [S_t][N_t][H_t], q);
    DECL_VLA_PTR_PT(T, q_data_a, [S_t][HS_t], q_data);
    DECL_VLA_PTR_PT(T, query_w_a, [N_t][H_t], query_w);

    auto k = q_data.new_empty({B_t, S_t, N_t, H_t});                    /* [512, 764, 8, 32] */
    DECL_VLA_PTR_PT(T, k_a, [S_t][N_t][H_t], k);
    DECL_VLA_PTR_PT(T, m_data_a, [S_t][HS_t], m_data);
    DECL_VLA_PTR_PT(T, key_w_a, [N_t][H_t], key_w);

    auto v = q_data.new_empty({B_t, S_t, N_t, H_t});                    /* [512, 764, 8, 32] */
    DECL_VLA_PTR_PT(T, v_a, [S_t][N_t][H_t], v);
    DECL_VLA_PTR_PT(T, value_w_a, [N_t][H_t], value_w);

    // auto logits = q_data.new_empty({B_t, N_t, S_t, S_t});               /* [512, 8, 764, 764] */
    // DECL_VLA_PTR_PT(T, logits_a, [N_t][S_t][S_t], logits);
    DECL_VLA_PTR_PT(T, bias_a, [1][1][S_t], bias);
    DECL_VLA_PTR_PT(T, nonbatched_bias_a, [N_t][S_t][S_t], nonbatched_bias);

    auto weights = q_data.new_empty({B_t, N_t, S_t, S_t});              /* [512, 8, 764, 764] */
    DECL_VLA_PTR_PT(T, weights_a, [N_t][S_t][S_t], weights);

    auto gate_values = q_data.new_empty({B_t, S_t, N_t, H_t});               /* [512, 764, 8, 32] */
    DECL_VLA_PTR_PT(T, gate_values_a, [S_t][N_t][H_t], gate_values);
    DECL_VLA_PTR_PT(T, gating_w_a, [N_t][H_t], gating_w);
    DECL_VLA_PTR_PT(T, gating_b_a, [H_t], gating_b);

    auto weighted_avg = q_data.new_empty({B_t, S_t, N_t, H_t});         /* [512, 764, 8, 32] */
    DECL_VLA_PTR_PT(T, weighted_avg_a, [S_t][N_t][H_t], weighted_avg);

    auto output = q_data.new_empty({B_t, S_t, HS_t});                   /* [512, 764, 256] */
    // DECL_VLA_PTR_PT(T, output_a, [S_t][HS_t], output);
    // DECL_VLA_PTR_PT(T, output_w_a, [H_t][HS_t], output_w);
    // DECL_VLA_PTR_PT(T, output_b_a, [1], output_b);
    // T* output_b_a = output_b.data_ptr<T>();
    

    int lda = HS_t;
    int ldb = N_t*H_t;
    int ldc = N_t*H_t;

    auto qkv_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(QKV_BLOCKSIZE, N_t*H_t, HS_t, 0, 0, lda, ldb, ldc, 0.0, 0, 1)));

    auto scale_tpp = SCOPEIT((ScaleTPP<T,T>(QKV_BLOCKSIZE*HS_t)), EW_SCL);
    auto zero_tpp = SCOPEIT(SetZeroTPP<T>(QKV_BLOCKSIZE*HS_t), EW_ZERO);
    float alpha = (1.0/sqrt(key_dim));

    // auto q = at::mul(at::einsum("bqa,ahc->bqhc", {q_data, query_w}), (1.0/sqrt(key_dim))) ;     /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8, 32] */
    {
    RECORD_SCOPE(alpha_q_gemm, {q, q_data, query_w});
        {
            RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
            #pragma omp parallel for collapse(2)
            for(int i=0; i < B_t; i++){
                for(int j=0; j < S_t; j += QKV_BLOCKSIZE){
                    T tmp[QKV_BLOCKSIZE][N_t][H_t];
                    zero_tpp(&tmp[0][0][0]);
                    qkv_brgemm_tpp(&q_data_a[i][j][0], &query_w_a[0][0][0], &tmp[0][0][0], 1);
                    scale_tpp(&tmp[0][0][0], &q_a[i][j][0][0], alpha);
                    // qkv_brgemm_tpp(&q_data_a[i][j][0], &query_w_a[0][0][0], &q_a[i][j][0][0], 1);
                    // scale_tpp(&q_a[i][j][0][0], &q_a[i][j][0][0], alpha);
                }
            }
        }
    }

    // auto k = at::einsum("bka,ahc->bkhc", {m_data, key_w});                                      /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8, 32] */
    {
    RECORD_SCOPE(alpha_k_gemm, {k, m_data, key_w});
        {
            RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
            #pragma omp parallel for collapse(2)
            for(int i=0; i < B_t; i++){
                for(int j=0; j < S_t; j += QKV_BLOCKSIZE){
                    qkv_brgemm_tpp(&m_data_a[i][j][0], &key_w_a[0][0][0], &k_a[i][j][0][0], 1);
                }
            }
        }
    }

    // auto v = at::einsum("bka,ahc->bkhc", {m_data, value_w});                                    /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8, 32] */
    {
    RECORD_SCOPE(alpha_v_gemm, {v, m_data, value_w});
        {
            RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
            #pragma omp parallel for collapse(2)
            for(int i=0; i < B_t; i++){
                for(int j=0; j < S_t; j += QKV_BLOCKSIZE){
                    qkv_brgemm_tpp(&m_data_a[i][j][0], &value_w_a[0][0][0], &v_a[i][j][0][0], 1);
                }
            }
        }
    }


    auto flag = nonbatched_bias.size(0) > 0;
    lda = H_t;
    ldb = A_BLOCKSIZE;
    ldc = S_t;
    
    auto a_trans_tpp = SCOPEIT(XformExtTPP<T>(A_BLOCKSIZE, H_t, H_t, A_BLOCKSIZE, N_t*H_t, A_BLOCKSIZE, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    auto a_cpy_tpp = SCOPEIT(CpyTPP<T>(A_BLOCKSIZE, H_t, N_t*H_t, H_t), EW_COPY);

    auto a_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(A_BLOCKSIZE, A_BLOCKSIZE, H_t, 0, 0, lda, ldb, ldc, 0.0, 0, 1)));
    // auto a_brgemm2_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(A_BLOCKSIZE, H_t, A_BLOCKSIZE, 0, 0, S_t, ldb, ldc, 0.0, 0, 1)));

    auto a_addbias_tpp = SCOPEIT(AddBiasTPP<T>(A_BLOCKSIZE, A_BLOCKSIZE, ldc), BIAS);
    auto a_add_nbbias_tpp = SCOPEIT((AddTPP<T,T>(A_BLOCKSIZE, A_BLOCKSIZE, ldc, ldc)), BIAS);

    auto a_softmax_tpp = SCOPEIT((VarSoftMaxFwdTPP<float,T>(A_BLOCKSIZE, S_t)), SOFTMAX);

    // logits = at::add(at::einsum("bqhc,bkhc->bhqk", {q, k}), bias);                         /* [512, 8, 764, 764]  = [512, 764, 8, 32] * [512, 764, 8, 32] + [512, 1, 1, 764] */
    // if (nonbatched_bias.size(0) > 0)
    //     logits = at::add(logits, at::unsqueeze(nonbatched_bias, 0));                       /* [512, 8, 764, 764]  = [512, 8, 764, 764] + [1, 8, 764, 764] */
    {
    RECORD_SCOPE(alpha_a_gemm, {q, k, bias});
        {
            RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
            #pragma omp parallel for collapse(2) 
            for(int i=0; i < B_t; i++){
                for(int j1=0; j1 < S_t; j1 += A_BLOCKSIZE){

                    T tmp_q[A_BLOCKSIZE * H_t];
                    T tmp_k[A_BLOCKSIZE * H_t];   
                    T tmp_logits[A_BLOCKSIZE][S_t];

                    for(int k=0; k < N_t; k++){

                        a_cpy_tpp(&q_a[i][j1][k][0], &tmp_q[0]);
                        
                        for(int j2=0; j2 < S_t; j2 += A_BLOCKSIZE){
                    
                            a_trans_tpp(&k_a[i][j2][k][0], &tmp_k[0]);                               // [A_BLOCKSIZE, 8, 32]  ----> [8, 32, A_BLOCKSIZE]

                            // a_brgemm_tpp(&tmp_q[0], &tmp_k[0], &logits_a[i][k][j1][j2], 1);
                            // a_addbias_tpp(&bias_a[i][0][0][j2], &logits_a[i][k][j1][j2]);
                            // if(flag)
                            //     a_add_nbbias_tpp(&nonbatched_bias_a[0][k][j1][j2], &logits_a[i][k][j1][j2], &logits_a[i][k][j1][j2]);

                            a_brgemm_tpp(&tmp_q[0], &tmp_k[0], &tmp_logits[0][j2], 1);
                            a_addbias_tpp(&bias_a[i][0][0][j2], &tmp_logits[0][j2]);
                            if(flag)
                                a_add_nbbias_tpp(&nonbatched_bias_a[0][k][j1][j2], &tmp_logits[0][j2], &tmp_logits[0][j2]);
                        }
                        a_softmax_tpp(1, &tmp_logits[0][0], &weights_a[i][k][j1][0]);
                    }
                }
            }
        }
    }
        
    // weights = at::_softmax(logits, -1, false);                                             /* [512, 8, 764, 764] = [512, 8, 764, 764] */

    weighted_avg = at::einsum("bhqk,bkhc->bqhc", {weights, v});                                              /* [512, 764, 8, 32]  = [512, 8, 764, 764] * [512, 764, 8, 32] */
    // DECL_VLA_PTR_PT(T, weighted_avg_a, [S_t][N_t][H_t], weighted_avg);

    lda = HS_t;
    ldb = N_t*H_t;
    ldc = N_t*H_t;

    auto c_zero_tpp = SCOPEIT(SetZeroTPP<T>(C_BLOCKSIZE, N_t*H_t, ldc), EW_ZERO);
    auto c_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(C_BLOCKSIZE, N_t*H_t, HS_t, 0, 0, lda, ldb, ldc, 0.0, 0, 1)));
    auto c_addbias_tpp = SCOPEIT(AddBiasTPP<T>(C_BLOCKSIZE, N_t*H_t, ldc), BIAS);
    auto c_sigmoid_tpp = SCOPEIT(SiLUFwdTPP<T>(C_BLOCKSIZE,  N_t * H_t, ldc, ldc), EW_MUL);

    auto c_mul_tpp = SCOPEIT((MulTPP<T,T>(C_BLOCKSIZE, N_t*H_t, ldc, ldc)), EW_MUL);

    // gate_values = at::sigmoid(at::add(at::einsum("bqc,chv->bqhv", {q_data, gating_w}), gating_b));           /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8, 32] + [8, 32]*/
    {
    RECORD_SCOPE(alpha_c_gemm, {weighted_avg, v, weights, q_data, gating_w, gating_b});
        {
            RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
            #pragma omp parallel for collapse(2)
            for(int i=0; i < B_t; i++){
                for(int j=0; j < S_t; j += C_BLOCKSIZE){
                    T tmp[C_BLOCKSIZE * N_t * H_t];
                    // T tmp_gate_values[C_BLOCKSIZE][N_t][H_t];
                    // c_zero_tpp(&tmp[0]);
                    c_brgemm_tpp(&q_data_a[i][j][0], &gating_w_a[0][0][0], &tmp[0], 1);
                    c_addbias_tpp(&gating_b_a[0][0], &tmp[0]);
                    c_sigmoid_tpp(&tmp[0], &tmp[0], &gate_values_a[i][j][0][0]);

                    // c_mul_tpp(&gate_values_a[i][j][0][0], &weighted_avg_a[i][j][0][0], &weighted_avg_a[i][j][0][0]);
                    // for(int b = 0; b < C_BLOCKSIZE; b++)
                    //     for(int n = 0; n < N_t; n++)
                    //         for(int h = 0; h < H_t; h++)
                    //             weighted_avg_a[i][j + b][n][h] *= (double)gate_values_a[i][j + b][n][h];
                }
            }

            
            // for(int i=0; i < B_t; i++)
            //     for(int j=0; j < S_t; j++)
            //         for(int n = 0; n < N_t; n++)
            //             for(int h = 0; h < H_t; h++)
            //                 weighted_avg_a[i][j][n][h] *= gate_values_a[i][j][n][h];
                    
    
        }
    }

    // weighted_avg = at::mul(weighted_avg, gate_values);                                                       /* [512, 764, 8, 32]  = [512, 764, 8, 32] * [512, 764, 8, 32] */
    weighted_avg.mul_(gate_values);

    output = at::add(at::einsum("bqhc,hco->bqo", {weighted_avg, output_w}), output_b);     /* [512, 764, 256]  = [512, 764, 8, 32] * [8, 32, 256] + [256] */
    
    // lda = N_t*H_t;
    // ldb = HS_t;
    // ldc = HS_t;
    
    // auto out_trans1_tpp = SCOPEIT(XformExtTPP<T>(N_t, H_t, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    // auto out_trans2_tpp = SCOPEIT(XformExtTPP<T>(HS_t, HS_t, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    // auto out_gemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(OUT_BLOCKSIZE, HS_t, N_t*H_t, 0, 0, lda, ldb, ldc, 0.0, 0, 1)));
    // auto out_addbias_tpp = SCOPEIT(AddBiasTPP<T>(OUT_BLOCKSIZE, HS_t, ldc), BIAS);
    
    // // output = at::einsum("bqhc,hco->bqo", {weighted_avg, output_w});

    // DECL_VLA_PTR_PT(T, output_a, [S_t][HS_t], output);
    // DECL_VLA_PTR_PT(T, weighted_avg_a, [S_t][N_t][H_t], weighted_avg);
    // DECL_VLA_PTR_PT(T, output_w_a, [H_t][HS_t], output_w);
    // DECL_VLA_PTR_PT(T, output_b_a, [1], output_b);

    // // output = at::add(output, output_b);
    // {
    // RECORD_SCOPE(output_linear, {output, weighted_avg, output_w, output_b});
    //     {
    //         // T tmp_w1[N_t * H_t * HS_t];
    //         // T tmp_w2[N_t * H_t * HS_t];
    //         // T tmp_w3[N_t * H_t * HS_t];

    //         // out_trans2_tpp(&output_w_a[0][0][0], &tmp_w1[0]);                           // [8, 32, 256] ---->  [256, 8, 32]
    //         // for (int k =0; k < HS_t; k++)                                              // [256, 8, 32] ---->  [256, 32, 8]
    //         //     out_trans1_tpp(&tmp_w1[k*N_t*H_t], &tmp_w2[k*N_t*H_t]);
            
    //         // out_trans2_tpp(&tmp_w2[0], &tmp_w3[0]);                                      // [256, 32, 8] ---->  [32, 8, 256]

    //         RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    //         #pragma omp parallel for collapse(2)
    //         for(int i=0; i < B_t; i++){
    //             for(int j=0; j < S_t; j += OUT_BLOCKSIZE){
                    
    //                 // T tmp[OUT_BLOCKSIZE][H_t][N_t];
    //                 // for (int k =0; k < OUT_BLOCKSIZE; k++)
    //                 //     out_trans1_tpp(&weighted_avg_a[i][j + k][0][0], &tmp[k][0][0]);

    //                 out_gemm_tpp(&weighted_avg_a[i][j][0][0], &output_w_a[0][0][0], &output_a[i][j][0], 1);
    //                 out_addbias_tpp(&output_b_a[0][0], &output_a[i][j][0]);
    //             }
    //         }
    //     }
    // }

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