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

/***********************************************************/
/* Helper functions for convolutions' general param setup */
/**********************************************************/

LIBXSMM_API_INLINE void  libxsmm_dnn_conv_get_feature_map_blocks( int C, int K, int* C_block, int* K_block, int* fm_lp_block, libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_blasint bc, libxsmm_blasint bk ) {
  int ifmblock = 0;
  int ofmblock = 0;
  int lp_block = 0;
  int tmp_max_c_block = bc;
  int tmp_max_k_block = bk;
  int tmp_block = 0;

  /* init libxsmm */
  LIBXSMM_INIT

  /* C */
  if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_VL256_CLX) || (libxsmm_target_archid >= LIBXSMM_X86_AVX512_VL256_CPX)
          || (libxsmm_target_archid == LIBXSMM_X86_AVX512_VL256)
        ){
    tmp_max_c_block = LIBXSMM_BLOCK_SIZE;
  } else if ( /*((libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR) && (datatype_in == LIBXSMM_DATATYPE_BF16)) ||*/
       (libxsmm_target_archid < LIBXSMM_X86_AVX512 ) ) {
    tmp_max_c_block = 32;
  } else if ( libxsmm_target_archid == LIBXSMM_AARCH64_V81 ) {
    tmp_max_c_block = 16;
  }
  if ( C <= tmp_max_c_block ) {
    ifmblock = C;
  } else if (C % tmp_max_c_block == 0) {
    ifmblock = tmp_max_c_block;
  } else {
    for ( tmp_block = 1; tmp_block <= tmp_max_c_block; tmp_block *= 2 ) {
      if ( C % tmp_block == 0 ) ifmblock = tmp_block;
    }
  }

  /* K */
  if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_VL256_CLX) || (libxsmm_target_archid >= LIBXSMM_X86_AVX512_VL256_CPX)
        || (libxsmm_target_archid == LIBXSMM_X86_AVX512_VL256)
      ){
    tmp_max_k_block = LIBXSMM_BLOCK_SIZE;
  } else if ( /*((libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR) && (datatype_in == LIBXSMM_DATATYPE_BF16)) ||*/
       (libxsmm_target_archid < LIBXSMM_X86_AVX512 ) ) {
    tmp_max_k_block = 32;
  } else if ( libxsmm_target_archid == LIBXSMM_AARCH64_V81 ) {
    tmp_max_k_block = 16;
  }
  if ( K <= tmp_max_k_block ) {
    ofmblock = K;
  } else if (K % tmp_max_k_block == 0) {
    ofmblock = tmp_max_k_block;
  } else {
    for ( tmp_block = 1; tmp_block <= tmp_max_k_block; tmp_block *= 2 ) {
      if ( K % tmp_block == 0 ) ofmblock = tmp_block;
    }
  }

  /* when do we need VNNI format? */
  if ( (datatype_in == LIBXSMM_DATATYPE_F32) && (datatype_out == LIBXSMM_DATATYPE_F32) ) {
    lp_block = 1;
  } else if ( (datatype_in == LIBXSMM_DATATYPE_BF16) && (datatype_out == LIBXSMM_DATATYPE_BF16) ) {
    lp_block = 2;
  } else if ( (datatype_in == LIBXSMM_DATATYPE_I16) && ((datatype_out == LIBXSMM_DATATYPE_I32) || (datatype_out == LIBXSMM_DATATYPE_F32)) ) {
    lp_block = 2;
  } else if (datatype_in == LIBXSMM_DATATYPE_I8) {
    lp_block = 4;
  } else {
    return;
  }

  *C_block = ifmblock;
  *K_block = ofmblock;
  *fm_lp_block = lp_block;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_ifmblock( conv_config* cfg ) {
  int result = 1;
  int ofm, lp;

  libxsmm_dnn_conv_get_feature_map_blocks( cfg->C, cfg->K, &result, &ofm, &lp, cfg->datatype_in, cfg->datatype_out, cfg->bc, cfg->bk );

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_ofmblock( conv_config* cfg ) {
  int result = 1;
  int ifm, lp;

  libxsmm_dnn_conv_get_feature_map_blocks( cfg->C, cfg->K, &ifm, &result, &lp, cfg->datatype_in, cfg->datatype_out, cfg->bc, cfg->bk );

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fm_lp_block( conv_config* cfg ) {
  int result = 1;
  int ifm, ofm;

  libxsmm_dnn_conv_get_feature_map_blocks( cfg->C, cfg->K, &ifm, &ofm, &result, cfg->datatype_in, cfg->datatype_out, cfg->bc, cfg->bk);

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fallback_loops_fwd( conv_config* cfg ) {
  int result = 0;
  /* FIXME: For now fallback only if MB is not divisible by number of threads */
  if (cfg->N % cfg->threads != 0) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_blocksifm( conv_config* cfg ) {
  int result = cfg->C / cfg->ifmblock;
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_blocksofm( conv_config* cfg ) {
  int result = cfg->K / cfg->ofmblock;
  return result;
}

/**********************************************************/
/* Helper functions for FWD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_ofw_rb( conv_config* cfg ) {
  int result = 0;
  result = cfg->ofw;
  if (cfg->ofw == 56) {
    result = 28;
  }
  if (cfg->datatype_in == LIBXSMM_DATATYPE_I8) {
    if (cfg->ofw % 2 == 0) {
      result = cfg->ofw/2;
    }
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_pack_input_fwd( conv_config* cfg ) {
  int result = 0;
  /* Pack only for small images and when having large K to amortize, and we can only pack for 1x1 convolutions */
  if ((cfg->ofw <= 14) && (cfg->K > 512) && (cfg->R == 1) && (cfg->S == 1) && (cfg->u == 2) && (cfg->v == 2)) {
    result = 1;
  }

#if 0
  /* For SPR we allow packing more aggressively to generate more efficient BRGEMMs */
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    if ((cfg->ofw <= 14) && (cfg->R == 1) && (cfg->S == 1) && (cfg->u == 2) && (cfg->v == 2)) {
      result = 1;
    }
  }
#endif

  /* Make sure we don't pack when minibatch is not divisible by number of threads since H is used potentially for parallelism */
  if (cfg->N != cfg->threads) {
    result = 0;
  }
  /* we don't pack for int8 */
  if (cfg->datatype_in == LIBXSMM_DATATYPE_I8) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_ofh_rb( conv_config* cfg ) {
  int result = 1;
  /* Multiple rows for "small" images and 1x1 convolutions */
  if ((cfg->ofh <= 14) && (cfg->R == 1) && (cfg->S == 1) && (cfg->pad_w_out == 0) && (cfg->pad_h_out == 0)) {
    result = cfg->ofh;
  }

  /* In this case we will be using fallback generic loops, thus ofh_rb should be 1 */
  if ((cfg->N % cfg->threads != 0) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) {
    result = 1;
  }

#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    if (cfg->ofw == 7 && cfg->ofh == 7 && cfg->R == 3 && cfg->S == 3) {
      result = 7;
    }
    if (cfg->ofw == 14 && cfg->ofh == 14 /*&& cfg->R == 3 && cfg->S == 3*/) {
      result = 2;
    }
  }
#endif

  /*  Make sure we don't use multiple rows when we don't pack input and convolutions are strided*/
  if ((cfg->pack_input == 0) && ((cfg->u !=1 ) || (cfg->v != 1))) {
    result = 1;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_pixels_gemm( conv_config* cfg ) {
  int result = cfg->fwd_ofw_rb * cfg->fwd_ofh_rb;
  /* In the case below we calculate redundantly pixels in order to efficiently use AMX */
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    if (cfg->R != 1 || cfg->S != 1) {
      if (cfg->ofw < 24) {
        result = (cfg->fwd_ofw_rb+2*cfg->pad_w) * (cfg->fwd_ofh_rb-2) + 2 * (cfg->fwd_ofw_rb+cfg->pad_w);
      }
    }
  }
#endif
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_block_H( conv_config* cfg ) {
  int result = 14;

#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    /* Spatial dimension block tuning for SPR */
    if ((cfg->ofh == 7 && cfg->u == 2) || (cfg->ofh == 14 && cfg->R != 3 ) ||  cfg->ofh == 27 || (cfg->ofh == 28 && cfg->R == 1) || cfg->ofh == 48 || cfg->ofh == 54 || cfg->ofh == 56 || cfg->ofh == 112 ) {
      result = 4;
    }
  } else {
    /* Block H only for large images  */
    if (cfg->ofh >= 28) {
      result = 4;
    }
    if (cfg->ofh == 28 && cfg->R == 3 ) {
      result = 14;
    }
  }
#else
  /* Block H only for large images  */
  if (cfg->ofh >= 28) {
    result = 4;
  }
  if (cfg->ofh == 28 && cfg->R == 3 ) {
    result = 14;
  }
#endif
  /* Make sure it is divisible bu the ofh_rb factor in the kernel */
  while ( result % cfg->fwd_ofh_rb != 0 ) {
    result--;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_blocksifm_blocking( conv_config* cfg ) {
  int result = 1;
  /* For 1x1 Convolutions bring in kernel all IFMs unless filters are huge*/
  if ((cfg->R == 1) && (cfg->S == 1) ) {
    result = cfg->blocksifm;
    if ((cfg->C >= 2048) && (cfg->K >= 512)) {
      result = 1;
    }
    if ( (cfg->target_archid < LIBXSMM_X86_AVX512_VL256) && (cfg->C >= 512) ) {
      result = 2;
    }
    if ( (cfg->target_archid < LIBXSMM_X86_AVX512_VL256) && (cfg->C >= 1024) ) {
      result = 4;
    }
  } else {
    result = 1;
    /* If small image can bring in more IFMS even if NOT 1x1 convolution */
    if (cfg->ofw <= 7) {
      result = 2;
    }
  }
  if (cfg->blocksifm % result != 0) {
    result = 1;
  }

  /* In case of SPR bring always in all accumulation */
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8))) {
    result = cfg->blocksifm;
  }
#endif

  if (cfg->datatype_in == LIBXSMM_DATATYPE_I8) {
    result = cfg->blocksifm;
  }

  if (cfg->datatype_in == LIBXSMM_DATATYPE_BF16) {
    result = cfg->blocksifm;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_loop_order_fwd( conv_config* cfg ) {
  int result = 0;
  /* Switch to loop order 1 only if 1x1 convolution with "large" input image and "small" K */
  if ((cfg->H >= 28) && (cfg->R == 1) && (cfg->S == 1) && (cfg->C >=512) && (cfg->K <=512)) {
    result = 1;
  }
  if (cfg->ofw == 56 && cfg->R == 1 && cfg->C == 256 && cfg->K == 64 ) {
    result = 1;
  }
  if (cfg->ofw == 28 && cfg->R == 1) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_fwd_IFM( conv_config* cfg ) {
  int result = 8;
  if (cfg->ofw == 7 && cfg->C == 2048 && cfg->K == 512) {
    result = 4;
  }
  /* Make sure it is divisible by ifms in the kernel  */
  while (result % cfg->blocksifm_blocking != 0) {
    result++;
  }
  result = LIBXSMM_MIN(cfg->blocksifm, result);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_fwd_OFM( conv_config* cfg ) {
  int result = 8;
  if (cfg->ofw == 14 && cfg->K == 1024) {
    result = 16;
  }
  if (cfg->ofw == 7) {
    result = 16;
  }
  result = LIBXSMM_MIN(cfg->blocksofm, result);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_use_ofm_parallelization( conv_config* cfg ) {
  int result = 0;
#if 0
  /* Use "hybrid" minibatch/ofm parallelization if we have huge filters */
  if ((cfg->R >= 3) && (cfg->S >= 3) && (cfg->C >= 512) && (cfg->K >= 512)) {
    result = 1;
  }
#endif
  if ((cfg->ofw <= 7) && (cfg->C == 1024) && (cfg->K == 512)) {
    result = 1;
  }
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8))) {
    if (cfg->ofw == 7) {
      result = 1;
    }
  }
#endif
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_rim_fmas_fwd( conv_config* cfg ) {
  int result = 0;
  /* Avoid rim FMA if the convolution is 3x3 (non-strided) and the image is "small" */
  if ((cfg->R == 3) && (cfg->S == 3) &&
      (cfg->u  == 1) && (cfg->v == 1) &&
      (cfg->pad_h_in == 1) && (cfg->pad_w_in == 1) &&
      (cfg->H == cfg->W) ) {
    if (cfg->ofw <= 28) {
      result = 1;
    }
    if (cfg->datatype_in == LIBXSMM_DATATYPE_I8) {
      result = 0;
    }
  }
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8))) {
    result = 0;
  }
#endif

  if (cfg->datatype_in == LIBXSMM_DATATYPE_BF16) {
    result = 0;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_shuffle_filter_accesses( conv_config* cfg ) {
  int result = 0;
  /* Shuffle filter accesses only if "pure minibatch" parallelization and large filters are involved */
  if ((cfg->use_ofm_parallelization == 0) && (cfg->C > 512) && (cfg->K > 512)) {
    result = 1;
  }
  if (cfg->ofw == 7 && cfg->R == 3 && cfg->C == 512) {
    result = 1;
  }
  if (cfg->ofw == 7 && cfg->R == 1 && cfg->C == 512 && cfg->K == 2048) {
    result = 1;
  }
  if (cfg->ofw == 7 && cfg->R == 1 && cfg->C == 2048 && cfg->K == 512) {
    result = 1;
  }
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) )  {
    result = 0;
  }
#endif
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_acc_load( conv_config* cfg ) {
  int result = 0;
  if ((cfg->overwrite_output) > 0) {
    if ((cfg->R == 1) && (cfg->S == 1)) {
      if (cfg->blocksifm_blocking == cfg->blocksifm) {
        result = 1;
      }
    } else {
      if ((cfg->blocksifm_blocking == cfg->blocksifm) && (cfg->avoid_fmas_in_rim == 0)) {
        result = 1;
      }
    }
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_init_fwd_gemm_flags( conv_config* cfg ) {
  int result = 0;

#if defined(LIBXSMM_DNN_CONV_SETUP_USE_NTS)
  /* If large image and NOT already loaded in accumulators, tnen use streaming stores */
  if ((cfg->ofw >= 56) && (cfg->K >= 256) && (cfg->avoid_acc_load == 1) && (cfg->R == 1) && (cfg->S == 1)) {
    result = LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }
  if (cfg->ofw == 56 && cfg->C == 64 && cfg->K == 64 && cfg->R == 1) {
    result = LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }
  if (cfg->ofw == 56 && cfg->C == 256 && cfg->K == 64 && cfg->R == 1) {
    result = LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }
  /* Disable since the GEMM output is going to f32 scratch  */
  if (cfg->datatype_in == LIBXSMM_DATATYPE_BF16 || cfg->datatype_in == LIBXSMM_DATATYPE_I8) {
    result = 0;
  }
#else
  LIBXSMM_UNUSED(cfg);
#endif
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8))) {
    result = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  }
#endif

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_padding_copy( conv_config* cfg ) {
  int result = 0;
  if ( (cfg->pad_h != cfg->pad_h_in) || (cfg->pad_w != cfg->pad_w_in) ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE void libxsmm_dnn_conv_setup_fwd_scratch( conv_config* cfg ) {
  cfg->fwd_packing_padding_scratch_size = 0;
  /* packing of input */
  if ( cfg->pack_input != 0 ) {
    cfg->fwd_packing_padding_scratch_size = (size_t)cfg->N * cfg->C *
      cfg->H/cfg->u *
      cfg->W/cfg->v *
      LIBXSMM_TYPESIZE(cfg->datatype_in);
  }
  /* logical padding with copying in the fly */
  if ( cfg->fwd_padding_copy != 0 ) {
    cfg->fwd_packing_padding_scratch_size = (size_t)cfg->N * cfg->C *
      (cfg->H + 2*cfg->pad_h) *
      (cfg->W + 2*cfg->pad_w) *
      LIBXSMM_TYPESIZE(cfg->datatype_in);
  }
  /* output buffer in high precision when we use BF16 */
  if ( ( cfg->datatype_in == LIBXSMM_DATATYPE_BF16 ) ||
      ( cfg->datatype_in == LIBXSMM_DATATYPE_I8 )      ) {
    cfg->fwd_lp_output_full_scratch_size = (size_t) LIBXSMM_MAX(cfg->threads * cfg->fwd_gemm_pixels * cfg->ofmblock * LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32), cfg->N * cfg->K * cfg->ofwp * cfg->ofhp * LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32));
    cfg->fwd_lp_output_block_scratch_size = (size_t)cfg->threads * cfg->fwd_ofw_rb *
      cfg->fwd_ofh_rb * cfg->ofmblock *
      LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32);
  } else {
    cfg->fwd_lp_output_full_scratch_size = 0;
    cfg->fwd_lp_output_block_scratch_size = 0;
  }
  /* align sizes to full cacheline */
  cfg->fwd_packing_padding_scratch_size += ( cfg->fwd_packing_padding_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->fwd_packing_padding_scratch_size % LIBXSMM_CACHELINE);
  cfg->fwd_lp_output_full_scratch_size += ( cfg->fwd_lp_output_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->fwd_lp_output_full_scratch_size % LIBXSMM_CACHELINE);
  cfg->fwd_lp_output_block_scratch_size += ( cfg->fwd_lp_output_block_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->fwd_lp_output_block_scratch_size % LIBXSMM_CACHELINE);

  /* set offsets */
  cfg->fwd_packing_padding_scratch_offset = 0;
  cfg->fwd_lp_output_full_scratch_offset = cfg->fwd_packing_padding_scratch_size;
  cfg->fwd_lp_output_block_scratch_offset = cfg->fwd_lp_output_full_scratch_offset +
    cfg->fwd_lp_output_full_scratch_size;

  /* set overall scratch size for forward */
  cfg->fwd_scratch_size = cfg->fwd_packing_padding_scratch_size +
    cfg->fwd_lp_output_full_scratch_size +
    cfg->fwd_lp_output_block_scratch_size;
}

/**********************************************************/
/* Helper functions for BWD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fallback_loops_bwd( conv_config* cfg ) {
  int result = 0;
  /* FIXME: Fallback if MB is not divisible by number of threads */
  if (cfg->N % cfg->threads != 0) {
    result = 1;
  }
  if (cfg->R == 1 && cfg->S == 1 && (cfg->pad_h != 0 ||  cfg->pad_w != 0)) {
    result = 1;
  }
  if ((cfg->R > 1 && cfg->pad_h == 0) || (cfg->S > 1 && cfg->pad_w == 0)) {
    result = 1;
  }
  if ((cfg->R > 1 && (cfg->pad_h_out == 0 || cfg->pad_h_in == 0)) ||
      (cfg->S > 1 && (cfg->pad_w_out == 0 || cfg->pad_w_in == 0))    ) {
    result = 1;
  }
  if ((cfg->R > 1 && cfg->u > 1) || (cfg->S > 1 && cfg->v > 1)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_bwd_ofw_rb( conv_config* cfg ) {
  int result = libxsmm_dnn_conv_setup_fwd_ofw_rb(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_bwd_ofh_rb( conv_config* cfg ) {
  int result = libxsmm_dnn_conv_setup_fwd_ofh_rb(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_bwd_pixels_gemm( conv_config* cfg ) {
  int result = cfg->bwd_ofw_rb * cfg->bwd_ofh_rb;
  /* In the case below we calculate redundantly pixels in order to efficiently use AMX */
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    if (cfg->R != 1 || cfg->S != 1) {
      if (cfg->ofw < 24) {
        result = (cfg->bwd_ofw_rb+2*cfg->pad_w) * (cfg->bwd_ofh_rb-2) + 2 * (cfg->bwd_ofw_rb+cfg->pad_w);
      }
    }
  }
#endif
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_bwd_block_H( conv_config* cfg ) {
  int result = 0;
  result = libxsmm_dnn_conv_setup_fwd_block_H(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_loop_order_bwd( conv_config* cfg ) {
  int result = 0;
  result = libxsmm_dnn_conv_setup_loop_order_fwd(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_bwd_IFM( conv_config* cfg ) {
  int result = 0;
  result = LIBXSMM_MIN(cfg->blocksifm, 16);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_bwd_OFM( conv_config* cfg ) {
  int result = 8;
  while (result % cfg->blocksofm_blocking != 0) {
    result++;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_pack_input_bwd( conv_config* cfg ) {
  int result = 0;
  if ((cfg->u != 1) && (cfg->bwd_ofh_rb != 1)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_use_ifm_parallelization( conv_config* cfg ) {
  int result = 0;
  if (cfg->ofw <= 7) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_rim_fmas_bwd( conv_config* cfg ) {
  int result = libxsmm_dnn_conv_setup_avoid_rim_fmas_fwd(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_blocksofm_blocking( conv_config* cfg ) {
  int result = 0;
  if (cfg->R == 1 && cfg->S == 1) {
    result = cfg->blocksofm;
  } else {
    result = 1;
    if (cfg->R == 3 && cfg->S == 3 && cfg->ofh == 7 && cfg->ofw == 7) {
      result = 2;
    }
  }
#if  0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    result = cfg->blocksofm;
  }
#endif

  if (cfg->blocksofm % result != 0) {
    result = 1;
  }

  if (cfg->datatype_in == LIBXSMM_DATATYPE_BF16) {
    result = cfg->blocksofm;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_init_bwd_gemm_flags( conv_config* cfg ) {
  int result = 0;
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    result = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  }
#endif
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_spread_input_bwd( conv_config* cfg ) {
  int result = 0;
  if (((cfg->u != 1) || (cfg->v != 1)) && (cfg->bwd_ofh_rb == 1)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_acc_load_bwd( conv_config* cfg ) {
  int result = 0;
  if (cfg->overwrite_output > 0) {
    if ((cfg->R == 1) && (cfg->S == 1)) {
      if (cfg->blocksofm_blocking == cfg->blocksofm) {
        result = 1;
      }
    } else {
      if ((cfg->blocksofm_blocking == cfg->blocksofm) && (cfg->avoid_fmas_in_rim == 0)) {
        result = 1;
      }
    }
  }
  return result;
}

LIBXSMM_API_INLINE void libxsmm_dnn_conv_setup_bwd_scratch( conv_config* cfg ) {
  /* transpose of weights */
  cfg->bwd_filter_trans_scratch_size = (size_t)cfg->C * cfg->K *
    cfg->R * cfg->S *
    LIBXSMM_TYPESIZE(cfg->datatype_in);

  cfg->bwd_packing_padding_scratch_size = 0;
  /* packing of input */
  if ( cfg->pack_input_bwd != 0 ) {
    cfg->bwd_packing_padding_scratch_size = (size_t)cfg->N * cfg->C *
      cfg->ofhp * cfg->ofwp *
      LIBXSMM_TYPESIZE(cfg->datatype_in);
  }
  /* logical padding with copying in the fly */
  if ( cfg->use_fallback_bwd_loops != 0 ) {
    cfg->bwd_packing_padding_scratch_size = (size_t)cfg->threads * cfg->ifmblock *
      (cfg->H + 2*cfg->pad_h) *
      (cfg->W + 2*cfg->pad_w) *
      LIBXSMM_TYPESIZE(cfg->datatype_in);
  }
  /* input bufffer in high precision when we use BF16 */
  if ( cfg->datatype_in == LIBXSMM_DATATYPE_BF16 ) {
    cfg->bwd_lp_input_full_scratch_size = (size_t) LIBXSMM_MAX(cfg->threads * cfg->bwd_gemm_pixels * cfg->ifmblock * LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32), cfg->N * cfg->C * cfg->ifwp * cfg->ifhp * LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32));
    /* logical padding with copying in the fly */
    if ( cfg->use_fallback_bwd_loops != 0 ) {
      cfg->bwd_packing_padding_scratch_size = (size_t)cfg->threads * cfg->ifmblock *
        (cfg->H + 2*cfg->pad_h) *
        (cfg->W + 2*cfg->pad_w) *
        LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32);
    }
  } else {
    cfg->bwd_lp_input_full_scratch_size = 0;
  }
  /* align sizes to full cacheline */
  cfg->bwd_filter_trans_scratch_size += ( cfg->bwd_filter_trans_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->bwd_filter_trans_scratch_size % LIBXSMM_CACHELINE);
  cfg->bwd_packing_padding_scratch_size += ( cfg->bwd_packing_padding_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->bwd_packing_padding_scratch_size % LIBXSMM_CACHELINE);
  cfg->bwd_lp_input_full_scratch_size += ( cfg->bwd_lp_input_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->bwd_lp_input_full_scratch_size % LIBXSMM_CACHELINE);

  /* set offsets */
  cfg->bwd_filter_trans_scratch_offset = 0;
  cfg->bwd_packing_padding_scratch_offset = cfg->bwd_filter_trans_scratch_size;
  cfg->bwd_lp_input_full_scratch_offset = cfg->bwd_packing_padding_scratch_offset +
    cfg->bwd_packing_padding_scratch_size;

  /* set overall scratch size for forward */
  cfg->bwd_scratch_size = cfg->bwd_filter_trans_scratch_size +
    cfg->bwd_packing_padding_scratch_size +
    cfg->bwd_lp_input_full_scratch_size;
}

/**********************************************************/
/* Helper functions for UPD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_weight_copies_upd( conv_config* cfg ) {
  int result = cfg->threads;
  if (cfg->ofw <= 14) {
    result = 9;
  }
  if (cfg->ofw == 14 && cfg->N == 92 && cfg->threads == 92) {
    result = 23;
  }
  if (cfg->ofw == 7 && cfg->N == 92 && cfg->threads == 92 && cfg->R == 3 && cfg->S == 3 && cfg->u == 1 && cfg->v == 1) {
    result = 23;
  }
  while (cfg->threads % result != 0) {
    result--;
  }
  /* FIXME: Hardcoded logic for N=27, N=26 */
  if (cfg->N == 27 && cfg->threads == 27 && cfg->R == 1 && cfg->ofw == 14 && cfg->u == 1) {
    result = 7;
  }
  if (((cfg->ofh == 14) || (cfg->ofw == 7 && cfg->u == 2)) && cfg->N == 26 && cfg->threads == 26) {
    result = 13;
  }
  if ((cfg->N != cfg->threads) && !(cfg->upd_linearized_tasklist == 0 && cfg->upd_use_batchreduce == 0)) {
    result = cfg->N;
  }
  /* Make sure a single copy when we use linearized-task view */
  if (cfg->upd_linearized_tasklist == 1) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE void libxsmm_dnn_conv_setup_bf16_upd_algorithms( conv_config* inout_cfg ) {
  conv_config res = *inout_cfg;
  int remainder_pixels, max_init_offset, max_compute_offset_input, input_compute_pad, accum_length_pixels, compute_pixels;
  const int multiple_target = 2;
  int IFHP = (res.upd_padding_copy == 1) ? res.ifhp + 2 * res.pad_h : res.ifhp;
  int IFWP = (res.upd_padding_copy == 1) ? res.ifwp + 2 * res.pad_w : res.ifwp;
  int OFHP = (res.upd_padding_copy == 1) ? res.ofhp + 2 * res.pad_h : res.ofhp;
  int OFWP = (res.upd_padding_copy == 1) ? res.ofwp + 2 * res.pad_w : res.ofwp;
  res.ifwp_extended = IFWP;
  res.upd_linearized_pixels = 1;
  if (res.S != 1 && res.v != 1) {
    res.upd_linearized_pixels = 0;
    res.upd_trans_w_only = 0;
  }
  if ((res.S != 1 && res.pad_w == 0) ||
      (res.R != 1 && res.pad_h == 0) ) {
    res.upd_linearized_pixels = 0;
    res.upd_trans_w_only = 0;
  }

  /* For large images facilitate the "large" transposes by blocking the pixel/reduction domains  */
  if (res.ofw >= 56 && res.ofh >=56 && res.R == 1 && res.S == 1 && res.u == 1 && res.v == 1) {
    res.upd_linearized_pixels = 0;
    res.upd_trans_w_only = 1;
  }

  res.on_the_fly_input_packing = 0;
  res.upd_pack_input_upfront = 0;
  res.use_hybrid_imgofm_parallelization = 0;
  res.upd_linearized_tasklist = 0;

  if (res.upd_linearized_pixels == 1) {
    /* Logistics to pad accumulation chainlength */
    compute_pixels = res.ofw * res.ofh + 2 * res.pad_w * (res.ofh-1);
    remainder_pixels = (compute_pixels % multiple_target == 0) ? 0 : (compute_pixels/multiple_target+1)*multiple_target - compute_pixels;
    accum_length_pixels = compute_pixels + remainder_pixels;

    /* In this case compact input upfront */
    if (res.R == 1 && res.S == 1 && (res.u != 1 || res.v != 1)) {
      res.upd_pack_input_upfront = 1;
    }

    /* Logistics for input transpose and additional pixel padding */
    max_init_offset = 2 * res.pad_h * IFWP + 2 * res.pad_w;
    max_compute_offset_input = max_init_offset + accum_length_pixels;
    input_compute_pad = (max_compute_offset_input > IFWP*IFHP) ? max_compute_offset_input - IFWP*IFHP : 0;
    res.input_pixels = IFWP * IFHP + input_compute_pad;
    if (res.upd_pack_input_upfront) {
      res.input_pixels = accum_length_pixels;
    }
    res.output_pixels = accum_length_pixels;
    res.pixel_blocking = accum_length_pixels;
    res.n_used_pixels = accum_length_pixels;
    res.compute_pixels = compute_pixels;

    res.use_intermediate_f32_wt_tensor = (res.pixel_blocking == res.n_used_pixels) ? 0 : 1;

    if (res.ofw <= 14) {
      res.use_hybrid_imgofm_parallelization = 1;
      res.weight_copies = libxsmm_dnn_conv_setup_weight_copies_upd(&res);
      if (res.ofw == 14 && res.K >= 1024) {
        res.use_hybrid_imgofm_parallelization = 0;
        res.weight_copies = res.threads;
      }
    } else {
      res.weight_copies = res.threads;
    }
  }

  if (res.upd_linearized_pixels == 0) {
    res.weight_copies = res.threads;
    if (res.v !=1) {
      res.on_the_fly_input_packing = 1;
    }
    remainder_pixels = (res.ofw % multiple_target == 0) ? 0 : (res.ofw/multiple_target+1)*multiple_target - res.ofw;
    res.ofwp_extended = OFWP + remainder_pixels;
    res.ifwp_extended = IFWP + remainder_pixels;
    if (res.ifwp_extended % 2 == 1) {
      res.ifwp_extended = res.ifwp_extended + 1;
    }
    res.output_pixels = OFHP * res.ofwp_extended;
    /* coverity[identical_branches] */
    res.batchreduce_h_pixels = (res.upd_trans_w_only) ? 1 : 1; /* TODO: identical_branches */
    res.use_intermediate_f32_wt_tensor = (res.batchreduce_h_pixels == res.ofh) ? 0 : 1;
  }

  if (res.N != res.threads) {
    res.use_intermediate_f32_wt_tensor = 1;
    res.use_hybrid_imgofm_parallelization = 0;
    res.weight_copies = LIBXSMM_MIN(res.N, res.threads);
  }

  *inout_cfg = res;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_loop_order_upd( conv_config* cfg ) {
  int result = 1;
  if (cfg->ofh == 28 && cfg->R == 1 && cfg->u == 1 && cfg->C == 128 && cfg->K == 512) {
    result = 0;
  }
  if (cfg->ofh == 28 && cfg->R == 3 && cfg->u == 1 && cfg->C == 128 && cfg->K == 128) {
    result = 0;
  }
  if (cfg->ofw == 28 && cfg->R == 1 && cfg->C == 256 && cfg->K == 512) {
    result = 0;
  }
  if (cfg->ofw == 14 && !(cfg->R == 1 && cfg->C == 1024 && cfg->K == 256)) {
    result = 0;
  }
  if (cfg->ofw == 7) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_pack_input_upd( conv_config* cfg ) {
  int result = 0;
  /* Pack input only for very small images, 1x1 convs, with large K to amortize the relevant overhead */
  if ((cfg->ofh <= 7) && (cfg->R == 1) && (cfg->S == 1) && (cfg->u != 1) && (cfg->v != 1) && (cfg->K >= 2048)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_rim_fmas_upd( conv_config* cfg ) {
  int result = 0;
  /* Avoid rim FMAs only for small images  */
  if ( (cfg->ofh <= 7) && (cfg->R == 3) && (cfg->S == 3) && (cfg->pad_w == 1) && (cfg->pad_h == 1)) {
    result = 1;
  }
  if (cfg->N != cfg->threads) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_upd_ofw_rb( conv_config* cfg ) {
  int result = 1;
  result = cfg->ofw;
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_upd_ofh_rb( conv_config* cfg ) {
  int result = 1;
  /* Restrict the reduction chain which is ofw_rb*ofh_rb*/
  if (cfg->ofh <= 28 ) {
    result = cfg->ofh;
  }
  /* In the following scenario with strided convolutions and non batch reduce kernel make sure we have ofh_rb = 1  */
  if ((cfg->u != 1) && (cfg->v != 1) && (cfg->upd_use_batchreduce == 0) && (cfg->upd_pack_input == 0)) {
    result = 1;
  }
  /* If using linearized taskview and have strided convs, make sure ofh_rb is 1.. */
  if (cfg->upd_linearized_tasklist == 1 && cfg->upd_avoid_rim_fmas == 0 && cfg->upd_pack_input == 0 && cfg->u != 1) {
    result = 1;
  }
  if (cfg->upd_linearized_tasklist == 1 && cfg->upd_use_batchreduce == 0 && (cfg->R != 1 || cfg->S != 1)) {
    result = 1;
  }
  if (cfg->upd_linearized_tasklist == 0 && cfg->upd_use_batchreduce == 0 && (cfg->R != 1 || cfg->S != 1)) {
    result = 1;
  }
  if (cfg->ofw == 56 && cfg->R == 1) {
    result = 2;
  }
  if (cfg->upd_linearized_tasklist == 1 && cfg->upd_use_batchreduce == 1 && cfg->upd_avoid_rim_fmas == 1) {
    result = cfg->ofh;
  }

  if ((cfg->N != cfg->threads) && (cfg->R > 1 || cfg->S > 1 ) && (cfg->u > 1 || cfg->v > 1 )) {
    result = 1;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_upd_IFM( conv_config* cfg ) {
  int result = 1;
  if (cfg->ofh == 56 && cfg->R == 1 && cfg->S == 1 && cfg->u == 1 && cfg->v == 1) {
    result = 4;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_upd_OFM( conv_config* cfg ) {
  int result = 1;
  LIBXSMM_UNUSED(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_img_batchreduce_block( conv_config* cfg ) {
  int result = 1;
  LIBXSMM_UNUSED(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_use_batchreduce_upd( conv_config* cfg ) {
  int result = 1;
  /* If W is large, no need for batchreduce kernel */
  if (cfg->ofw >= 56) {
    result = 0;
  }
  /* If we have packed the input, then disable batch-reduce GEMM */
  if (cfg->upd_pack_input == 1) {
    result = 0;
  }
  if (cfg->upd_linearized_tasklist == 1 && cfg->upd_avoid_rim_fmas == 0) {
    result = 0;
  }
  if (cfg->upd_linearized_tasklist == 1 && cfg->upd_avoid_rim_fmas == 1) {
    result = 1;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_linearized_tasklist_upd( conv_config* cfg ) {
  int result = 0;
  /* Use linearized task-list (i.e. no reduction) only if small images and large filters */
  if (cfg->ofh <= 10 && cfg->ofw <= 10) {
    result = 1;
  }
  if (cfg->ofw == 7 && cfg->N == 92 && cfg->threads == 92 && cfg->R == 3 && cfg->S == 3 && cfg->u == 1 && cfg->v == 1) {
    result = 0;
  }
  if (cfg->ofh == 14  && cfg->ofw == 14 && cfg->N == 23 && cfg->threads == 23) {
    result = 1;
  }
#if 0
  if ((cfg->blocksofm * cfg->blocksifm * cfg->R * cfg->S > (cfg->threads * 4)) && (cfg->ofh <= 56)) {
    result = 1;
  }
#endif
  if (cfg->u == 2 && cfg->v == 2 && cfg->K == 512) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_init_upd_gemm_flags( conv_config* cfg ) {
  int result = 0;
  LIBXSMM_UNUSED(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_upd_padding_copy( conv_config* cfg ) {
  int result = 0;
  if ( (cfg->pad_h != cfg->pad_h_in) || (cfg->pad_w != cfg->pad_w_in) ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE void libxsmm_dnn_conv_setup_upd_scratch( conv_config* cfg ) {
  cfg->upd_packing_padding_scratch_size = 0;
  /* packing of input */
  if ( cfg->upd_pack_input != 0 ) {
    cfg->upd_packing_padding_scratch_size = (size_t)cfg->N * cfg->C *
      cfg->H/cfg->u *
      cfg->W/cfg->v *
      LIBXSMM_TYPESIZE(cfg->datatype_in);
  }
  /* logical padding with copying in the fly */
  if ( cfg->upd_padding_copy != 0 ) {
    cfg->upd_packing_padding_scratch_size = (size_t)cfg->N * cfg->C *
      (cfg->H + 2*cfg->pad_h) *
      (cfg->W + 2*cfg->pad_w) *
      LIBXSMM_TYPESIZE(cfg->datatype_in);
  }
  /* output/input buffer to transpose when we use bf16 */
  if ( cfg->datatype_in == LIBXSMM_DATATYPE_BF16 ) {
#if 0
    if  (cfg->target_archid >= LIBXSMM_X86_AVX512_SPR) {
      int OFHP = (cfg->upd_padding_copy == 1) ? cfg->ofhp + 2 * cfg->pad_h : cfg->ofhp;
      int IFHP = (cfg->upd_padding_copy == 1) ? cfg->ifhp + 2 * cfg->pad_h : cfg->ifhp;

      if (cfg->upd_linearized_pixels == 1) {
        cfg->upd_lp_output_full_scratch_size = (size_t) (cfg->N * cfg->output_pixels * cfg->K * sizeof(cfg->datatype_in));
        cfg->upd_lp_input_full_scratch_size = (size_t) (cfg->N * cfg->input_pixels * cfg->C * sizeof(cfg->datatype_in));
      }

      if (cfg->upd_linearized_pixels == 0) {
        cfg->upd_lp_output_full_scratch_size = (size_t) (cfg->N * OFHP * cfg->ofwp_extended * cfg->K * sizeof(cfg->datatype_in));
        cfg->upd_lp_input_full_scratch_size = (size_t) (cfg->N * IFHP * cfg->ifwp_extended * cfg->C * sizeof(cfg->datatype_in));
      }
    } else {
#endif
    if (1) {
      const int multiple_target = 2;
      int IFHP = (cfg->upd_padding_copy == 1) ? cfg->ifhp + 2 * cfg->pad_h : cfg->ifhp;
      int IFWP = (cfg->upd_padding_copy == 1) ? cfg->ifwp + 2 * cfg->pad_w : cfg->ifwp;
      int OFHP = (cfg->upd_padding_copy == 1) ? cfg->ofhp + 2 * cfg->pad_h : cfg->ofhp;
      int OFWP = (cfg->upd_padding_copy == 1) ? cfg->ofwp + 2 * cfg->pad_w : cfg->ofwp;

      if (cfg->upd_linearized_pixels == 1) {
        int compute_pixels = cfg->ofw * cfg->ofh + 2 * cfg->pad_w * (cfg->ofh-1);
        int remainder_pixels = (compute_pixels % multiple_target == 0) ? 0 : (compute_pixels/multiple_target+1)*multiple_target - compute_pixels;
        int accum_length_pixels = compute_pixels + remainder_pixels;

        int max_init_offset = 2 * cfg->pad_h * IFWP + 2 * cfg->pad_w;
        int max_compute_offset_input = max_init_offset + accum_length_pixels;
        int input_compute_pad = (max_compute_offset_input > IFWP*IFHP) ? max_compute_offset_input - IFWP*IFHP : 0;
        int input_pixels = IFWP * IFHP + input_compute_pad;

        if (cfg->upd_pack_input_upfront == 1) {
          input_pixels = accum_length_pixels;
        }

        cfg->upd_lp_output_full_scratch_size = (size_t) (cfg->N * accum_length_pixels * cfg->K * sizeof(cfg->datatype_in));
        cfg->upd_lp_input_full_scratch_size = (size_t) (cfg->N * input_pixels * cfg->C * sizeof(cfg->datatype_in));
      }

      if (cfg->upd_linearized_pixels == 0) {
        int remainder_pixels = (cfg->ofw % multiple_target == 0) ? 0 : (cfg->ofw/multiple_target+1)*multiple_target - cfg->ofw;
        int ofwp_extended = OFWP + remainder_pixels;
        int ifwp_extended = IFWP + remainder_pixels;

        cfg->upd_lp_output_full_scratch_size = (size_t) (cfg->N * OFHP * ofwp_extended * cfg->K * sizeof(cfg->datatype_in));
        cfg->upd_lp_input_full_scratch_size = (size_t) (cfg->N * IFHP * ifwp_extended * cfg->C * sizeof(cfg->datatype_in));
      }
    }
    cfg->upd_lp_filter_full_scratch_size = (size_t)cfg->R * cfg->S * cfg->C * cfg->K * cfg->threads *
      LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32);
  } else {
    cfg->upd_lp_output_full_scratch_size = 0;
    cfg->upd_lp_input_full_scratch_size = 0;
    cfg->upd_lp_filter_full_scratch_size = 0;
  }
  /* filter scratch */
  cfg->upd_filter_scratch_size = (size_t) cfg->R * cfg->S * cfg->C * cfg->K * LIBXSMM_MAX(cfg->threads, cfg->N) * sizeof(float);

  /* align sizes to full cacheline */
  cfg->upd_packing_padding_scratch_size += ( cfg->upd_packing_padding_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->upd_packing_padding_scratch_size % LIBXSMM_CACHELINE);
  cfg->upd_lp_output_full_scratch_size += ( cfg->upd_lp_output_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->upd_lp_output_full_scratch_size % LIBXSMM_CACHELINE);
  cfg->upd_lp_input_full_scratch_size += ( cfg->upd_lp_input_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->upd_lp_input_full_scratch_size % LIBXSMM_CACHELINE);
  cfg->upd_filter_scratch_size += ( cfg->upd_filter_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->upd_filter_scratch_size % LIBXSMM_CACHELINE);
  cfg->upd_lp_filter_full_scratch_size += ( cfg->upd_lp_filter_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->upd_lp_filter_full_scratch_size % LIBXSMM_CACHELINE);

  /* calculate offsets */
  cfg->upd_packing_padding_scratch_offset = 0;
  cfg->upd_lp_output_full_scratch_offset = cfg->upd_packing_padding_scratch_size;
  cfg->upd_lp_input_full_scratch_offset = cfg->upd_lp_output_full_scratch_offset + cfg->upd_lp_output_full_scratch_size;
  cfg->upd_filter_scratch_offset = cfg->upd_lp_input_full_scratch_offset + cfg->upd_lp_input_full_scratch_size;
  cfg->upd_lp_filter_full_scratch_offset = cfg->upd_filter_scratch_offset + cfg->upd_filter_scratch_size;

  /* set overall scratch size for update */
  cfg->upd_scratch_size = cfg->upd_packing_padding_scratch_size +
    cfg->upd_lp_output_full_scratch_size +
    cfg->upd_lp_input_full_scratch_size +
    cfg->upd_filter_scratch_size +
    cfg->upd_lp_filter_full_scratch_size;
}

LIBXSMM_API conv_config setup_conv_config( libxsmm_datatype cnn_dtype_in, libxsmm_datatype cnn_dtype_out, libxsmm_blasint N, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint R, libxsmm_blasint S,
    libxsmm_blasint stride_h, libxsmm_blasint stride_w,
    libxsmm_blasint pad_h, libxsmm_blasint pad_w,
    libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in,
    libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
    libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, libxsmm_dnn_conv_eltwise_fuse fuse_type, libxsmm_blasint overwrite_output, libxsmm_blasint avoid_bwd_wt_trans, libxsmm_blasint zero_fwd_output_rim) {
  conv_config res;

  /* printf("debug: calling setup_conv_config with N H W C K R S padding stride_h stride_w bc bk: %d %d %d %d %d %d %d | %d %d %d %d %d %d | %d %d | %d %d\n",
                                                                                    N, H, W, C, K, R, S,
                                                                                    pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out,
                                                                                    stride_h, stride_w, bc, bk); */

  memset(&res, 0, sizeof(conv_config));

  /* init libxsmm */
  LIBXSMM_INIT

  /* Generic parameter setup  */
  res.N = N;
  res.H = H;
  res.W = W;
  res.C = C;
  res.K = K;
  res.R = R;
  res.S = S;
  res.u = stride_h;
  res.v = stride_w;
  res.pad_h = pad_h;
  res.pad_w = pad_w;
  res.pad_h_in = pad_h_in;
  res.pad_w_in = pad_w_in;
  res.pad_h_out = pad_h_out;
  res.pad_w_out = pad_w_out;
  res.threads = threads;
  res.target_archid = libxsmm_target_archid;
  res.datatype_in   = cnn_dtype_in;
  res.datatype_out  = cnn_dtype_out;
  res.ifhp = res.H + 2*res.pad_h_in;
  res.ifwp = res.W + 2*res.pad_w_in;
  res.ofh = (res.H + 2*res.pad_h - res.R) / res.u + 1;
  res.ofw = (res.W + 2*res.pad_w - res.S) / res.v + 1;
  res.ofhp = res.ofh + 2*res.pad_h_out;
  res.ofwp = res.ofw + 2*res.pad_w_out;
  res.ifmblock = 1;
  res.ofmblock = 1;
  res.blocksifm = res.C;
  res.blocksofm = res.K;
  res.fwd_ofw_rb = 1;
  res.fwd_ofh_rb = 1;
  res.bwd_ofw_rb = 1;
  res.bwd_ofh_rb = 1;
  res.upd_ofw_rb = 1;
  res.upd_ofh_rb = 1;
  res.fm_lp_block = 1;
  res.blocksifm_blocking = 1;
  res.blocksofm_blocking = 1;
  res.avoid_bwd_wt_trans = avoid_bwd_wt_trans;
  res.overwrite_output   = overwrite_output;
  res.zero_fwd_output_rim= zero_fwd_output_rim;
  res.fuse_type          = fuse_type;
  res.bc = bc;
  res.bk = bk;

  /* Use helper functions to setup convolutions */
  res.ifmblock      = libxsmm_dnn_conv_setup_ifmblock(&res);
  res.ofmblock      = libxsmm_dnn_conv_setup_ofmblock(&res);
  res.fm_lp_block   = libxsmm_dnn_conv_setup_fm_lp_block(&res);
  res.blocksifm     = libxsmm_dnn_conv_setup_blocksifm(&res);
  res.blocksofm     = libxsmm_dnn_conv_setup_blocksofm(&res);

  /* FWD parameter setup  */
  res.fwd_ofw_rb              = libxsmm_dnn_conv_setup_fwd_ofw_rb(&res);
  res.pack_input              = libxsmm_dnn_conv_setup_pack_input_fwd(&res);
  res.fwd_ofh_rb              = libxsmm_dnn_conv_setup_fwd_ofh_rb(&res);
  res.fwd_gemm_pixels         = libxsmm_dnn_conv_setup_fwd_pixels_gemm(&res);
  res.block_fwd_oj            = libxsmm_dnn_conv_setup_fwd_block_H(&res);
  res.loop_order              = libxsmm_dnn_conv_setup_loop_order_fwd(&res);
  res.blocksifm_blocking      = libxsmm_dnn_conv_setup_blocksifm_blocking(&res);
  res.block_fwd_ofm           = libxsmm_dnn_conv_setup_block_fwd_OFM(&res);
  res.block_fwd_ifm           = libxsmm_dnn_conv_setup_block_fwd_IFM(&res);
  res.avoid_fmas_in_rim       = libxsmm_dnn_conv_setup_avoid_rim_fmas_fwd(&res);
  res.use_ofm_parallelization = libxsmm_dnn_conv_setup_use_ofm_parallelization(&res);
  res.shuffle_filter_accesses = libxsmm_dnn_conv_setup_shuffle_filter_accesses(&res);
  res.avoid_acc_load          = libxsmm_dnn_conv_setup_avoid_acc_load(&res);
  res.fwd_flags               = libxsmm_dnn_conv_setup_init_fwd_gemm_flags(&res);
  res.use_fallback_fwd_loops  = libxsmm_dnn_conv_setup_fallback_loops_fwd(&res);
  res.fwd_padding_copy        = libxsmm_dnn_conv_setup_fwd_padding_copy(&res);
  /* Generate FWD kernels  */
  //libxsmm_dnn_conv_generate_fwd_kernels(&res);

  /* BWD parameter setup  */
  res.bwd_ofw_rb = libxsmm_dnn_conv_setup_bwd_ofw_rb(&res);
  res.bwd_ofh_rb = libxsmm_dnn_conv_setup_bwd_ofh_rb(&res);
  res.bwd_gemm_pixels = libxsmm_dnn_conv_setup_bwd_pixels_gemm(&res);
  res.pack_input_bwd = libxsmm_dnn_conv_setup_pack_input_bwd(&res);
  res.spread_input_bwd = libxsmm_dnn_conv_setup_spread_input_bwd(&res);
  res.blocksofm_blocking = libxsmm_dnn_conv_setup_blocksofm_blocking(&res);
  res.avoid_acc_load_bwd = libxsmm_dnn_conv_setup_avoid_acc_load_bwd(&res);
  res.use_ifm_parallelization = libxsmm_dnn_conv_setup_use_ifm_parallelization(&res);
  res.block_bwd_ofm = libxsmm_dnn_conv_setup_block_bwd_OFM(&res);
  res.block_bwd_ifm = libxsmm_dnn_conv_setup_block_bwd_IFM(&res);
  res.block_bwd_oj = libxsmm_dnn_conv_setup_bwd_block_H(&res);
  res.use_fallback_bwd_loops = libxsmm_dnn_conv_setup_fallback_loops_bwd(&res);
  res.bwd_flags = libxsmm_dnn_conv_setup_init_bwd_gemm_flags(&res);
  /* Generate BWD kernels  */
  //libxsmm_dnn_conv_generate_bwd_kernels(&res);

  /* UPD parameter setup */
  res.upd_linearized_tasklist = libxsmm_dnn_conv_setup_linearized_tasklist_upd(&res);
  res.upd_avoid_rim_fmas = libxsmm_dnn_conv_setup_avoid_rim_fmas_upd(&res);
  res.upd_pack_input = libxsmm_dnn_conv_setup_pack_input_upd(&res);
  res.upd_use_batchreduce = libxsmm_dnn_conv_setup_use_batchreduce_upd(&res);
  res.upd_ofw_rb = libxsmm_dnn_conv_setup_upd_ofw_rb(&res);
  res.upd_ofh_rb = libxsmm_dnn_conv_setup_upd_ofh_rb(&res);
  res.upd_loop_order = libxsmm_dnn_conv_setup_loop_order_upd(&res);
  res.weight_copies = libxsmm_dnn_conv_setup_weight_copies_upd(&res);
  res.block_upd_ofm = libxsmm_dnn_conv_setup_block_upd_OFM(&res);
  res.block_upd_ifm = libxsmm_dnn_conv_setup_block_upd_IFM(&res);
  res.upd_loop_order = libxsmm_dnn_conv_setup_loop_order_upd(&res);
  res.upd_padding_copy = libxsmm_dnn_conv_setup_upd_padding_copy(&res);

  if (cnn_dtype_in == LIBXSMM_DATATYPE_BF16) {
    libxsmm_dnn_conv_setup_bf16_upd_algorithms(&res);
  }

  /* Generate UPD kernels  */
  //libxsmm_dnn_conv_generate_upd_kernels(&res);

  /* let's configure  scratch */
  libxsmm_dnn_conv_setup_fwd_scratch( &res );
  libxsmm_dnn_conv_setup_bwd_scratch( &res );
  libxsmm_dnn_conv_setup_upd_scratch( &res );
  res.scratch_size = res.fwd_scratch_size + res.bwd_scratch_size + res.upd_scratch_size;

  /* setting up the barrier */
  //res.barrier = libxsmm_barrier_create(threads, 1);

  return res;
}
