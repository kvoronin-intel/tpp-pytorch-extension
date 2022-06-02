#ifndef CONV_SETUP_EXTERNAL_H
#define CONV_SETUP_EXTERNAL_H

#define MIXED 0
#define KHWC 1
#define HWKC 2
#define CHWK 3
#define HWCK 4
#define  LIBXSMM_DNN_CONV_SETUP_USE_NTS

#define LIBXSMM_BLOCK64
#if defined LIBXSMM_BLOCK64
# define LIBXSMM_BLOCK_SIZE 64
#else
# define LIBXSMM_BLOCK_SIZE 32
#endif

typedef enum libxsmm_dnn_conv_eltwise_fuse {
  LIBXSMM_DNN_CONV_ELTWISE_FUSE_NONE = 0,
  LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS = 1,
  LIBXSMM_DNN_CONV_ELTWISE_FUSE_RELU = 2,
  LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS_RELU = LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS | LIBXSMM_DNN_CONV_ELTWISE_FUSE_RELU
} libxsmm_dnn_conv_eltwise_fuse;

typedef enum libxsmm_dnn_conv_pass {
  LIBXSMM_DNN_CONV_PASS_FWD   = 1,
  LIBXSMM_DNN_CONV_PASS_BWD_D = 2,
  LIBXSMM_DNN_CONV_PASS_BWD_W = 4,
  LIBXSMM_DNN_CONV_PASS_BWD   = 6
} libxsmm_dnn_conv_pass;

typedef struct conv_config {
  /* Convolution params  */
  libxsmm_blasint N;
  libxsmm_blasint H;
  libxsmm_blasint W;
  libxsmm_blasint C;
  libxsmm_blasint K;
  libxsmm_blasint R;
  libxsmm_blasint S;
  libxsmm_blasint u;
  libxsmm_blasint v;
  libxsmm_blasint pad_h;
  libxsmm_blasint pad_w;
  libxsmm_blasint pad_h_in;
  libxsmm_blasint pad_w_in;
  libxsmm_blasint pad_h_out;
  libxsmm_blasint pad_w_out;
  libxsmm_blasint threads;
  libxsmm_blasint  overwrite_output;
  libxsmm_blasint  avoid_bwd_wt_trans;
  libxsmm_blasint  zero_fwd_output_rim;
  libxsmm_dnn_conv_eltwise_fuse  fuse_type;
  libxsmm_datatype datatype_in;
  libxsmm_datatype datatype_out;
  int target_archid;

  /* additional size for internal data types */
  int bc;
  int bk;
  int ifhp;
  int ifwp;
  int ofh;
  int ofw;
  int ofhp;
  int ofwp;
  int ifmblock;
  int ofmblock;
  int blocksifm;
  int blocksofm;
  int fwd_ofw_rb;
  int fwd_ofh_rb;
  int bwd_ofw_rb;
  int bwd_ofh_rb;
  int upd_ofw_rb;
  int upd_ofh_rb;
  int fm_lp_block; /* additional blocking for low precision datatypes of feature maps */
  int blocksifm_blocking;
  int blocksofm_blocking;
  int avoid_acc_load;
  int avoid_acc_load_bwd;
  int pack_input;
  int pack_input_bwd;
  int spread_input_bwd;
  int weight_copies;
  int loop_order;
  int use_ofm_parallelization;
  int use_ifm_parallelization;
  int avoid_fmas_in_rim;
  int upd_use_batchreduce;
  int upd_pack_input;
  int upd_loop_order;
  int upd_linearized_tasklist;
  int upd_avoid_rim_fmas;
  int fwd_flags;
  int bwd_flags;
  int shuffle_filter_accesses;
  int use_fallback_fwd_loops;
  int use_fallback_bwd_loops;
  int fwd_gemm_pixels;
  int bwd_gemm_pixels;
  int input_pixels;
  int output_pixels;
  int n_used_pixels;
  int pixel_blocking;
  int use_intermediate_f32_wt_tensor;
  int upd_linearized_pixels;
  int ifwp_extended;
  int ofwp_extended;
  int batchreduce_h_pixels;
  int on_the_fly_input_packing;
  int upd_pack_input_upfront;
  int use_hybrid_imgofm_parallelization;
  int remainder_pixels;
  int pack_to_cnhw;
  int fuse_upd_transposes;
  int compute_pixels;
  int upd_trans_w_only;
  int fwd_padding_copy;
  int upd_padding_copy;
  int upd_remaining_pixels;
  int block_fwd_oj;
  int block_fwd_ifm;
  int block_fwd_ofm;
  int block_bwd_oj;
  int block_bwd_ifm;
  int block_bwd_ofm;
  int block_upd_ifm;
  int block_upd_ofm;

  /* scratch */
  size_t fwd_packing_padding_scratch_size;
  size_t fwd_lp_output_full_scratch_size;
  size_t fwd_lp_output_block_scratch_size;
  size_t fwd_packing_padding_scratch_offset;
  size_t fwd_lp_output_full_scratch_offset;
  size_t fwd_lp_output_block_scratch_offset;
  size_t fwd_scratch_size;

  size_t bwd_filter_trans_scratch_size;
  size_t bwd_packing_padding_scratch_size;
  size_t bwd_lp_input_full_scratch_size;
  size_t bwd_filter_trans_scratch_offset;
  size_t bwd_packing_padding_scratch_offset;
  size_t bwd_lp_input_full_scratch_offset;
  size_t bwd_scratch_size;

  size_t upd_packing_padding_scratch_size;
  size_t upd_lp_output_full_scratch_size;
  size_t upd_lp_input_full_scratch_size;
  size_t upd_filter_scratch_size;
  size_t upd_lp_filter_full_scratch_size;
  size_t upd_packing_padding_scratch_offset;
  size_t upd_lp_output_full_scratch_offset;
  size_t upd_lp_input_full_scratch_offset;
  size_t upd_lp_filter_full_scratch_offset;
  size_t upd_filter_scratch_offset;
  size_t upd_scratch_size;

  size_t scratch_size;

} conv_config;


extern conv_config conv_setup(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint K, libxsmm_blasint R, libxsmm_blasint S,
                              libxsmm_blasint pad_h, libxsmm_blasint pad_w, libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in, libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                              libxsmm_blasint stride, int dtype_int );

#endif /* CONV_SETUP_EXTERNAL_H */
