#include <sys/mman.h>

#include <torch/extension.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#pragma message "Using OpenMP"
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

#ifdef __x86_64__
#include <xmmintrin.h>
#endif

#include <iostream>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

#include <libxsmm.h>

#define PART_OF_EXTENSIONS

#ifdef PART_OF_EXTENSIONS
#include "init.h"
#endif

//#define POOLING_SCALAR_CODE

#ifdef POOLING_SCALAR_CODE
#  warning "POOLING_SCALAR_CODE is defined (terrible performance)"
#endif

//#define OLD_LIBXSMM_HANDLES

//#define TIMING

#ifdef TIMING

double getFreq();

double ifreq = 1.0 / getFreq();

#ifdef __x86_64__
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
#elif defined(__aarch64__)
static __inline__ unsigned long long rdtsc(void) {
  unsigned long long val;

  /*
   * According to ARM DDI 0487F.c, from Armv8.0 to Armv8.5 inclusive, the
   * system counter is at least 56 bits wide; from Armv8.6, the counter
   * must be 64 bits wide.  So the system counter could be less than 64
   * bits wide and it is attributed with the flag 'cap_user_time_short'
   * is true.
   */
  asm volatile("mrs %0, cntvct_el0" : "=r"(val));

  return val;
}
#else
#error "Unsupported architecture for rdtsc"
#endif
inline double getFreq() {
  long long int s = rdtsc();
  sleep(1);
  long long int e = rdtsc();
  return (e - s) * 1.0;
}

inline double getTime() {
  return rdtsc() * ifreq;
}

#endif

#define MB (1024.0*1024.0)
#define GB (1024.0*1024.0*1024.0)

#ifdef OLD_LIBXSMM_HANDLES
  #define CHKERR_LIBXSMM_DNN(A) { const int chkerr_libxsmm_dnn_ = A; if (LIBXSMM_DNN_SUCCESS != chkerr_libxsmm_dnn_) { \
    fprintf(stderr, "%s\n", libxsmm_dnn_get_error(chkerr_libxsmm_dnn_)); global_status = chkerr_libxsmm_dnn_; } \
  }
#endif /* OLD_LIBXSMM_HANDLES */

#if 1
# define PRINT_LAYOUT(DESC, LAYOUT, PT_TENSOR) print_layout(DESC, LAYOUT, PT_TENSOR)
#else
# define PRINT_LAYOUT(DESC, LAYOUT, PT_TENSOR)
#endif

#define NEW_BATCHNORM
#define NEW_GROUPNORM
#define NEW_CONV
#define NEW_FC
#define NEW_POOLING

#define NEW_BOTTLENECK

//#define RECORD_FUNCTIONS_MACRO


#if defined(NEW_BATCHNORM) || defined(NEW_GROUPNORM) || defined(NEW_CONV) || defined(NEW_FC) || defined(NEW_POOLING)
    /* for init_buf/zero_buf only? */
    #include "dnn_common.h"

    #include "libxsmm_dnn.h"


//    #include "libxsmm_dnn_conv_setup.h" /* specifically for libxsmm_dnn_conv_get_feature_map_blocks */
#ifndef PART_OF_EXTENSIONS

#define LIBXSMM_BLOCK64
#if defined LIBXSMM_BLOCK64
# define LIBXSMM_BLOCK_SIZE 64
#else
# define LIBXSMM_BLOCK_SIZE 32
#endif

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
#else
LIBXSMM_API_INLINE void  libxsmm_dnn_conv_get_feature_map_blocks( int C, int K, int* C_block, int* K_block, int* fm_lp_block, libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_blasint bc, libxsmm_blasint bk );
#endif /* ifndef-else PART_OF_EXTENSIONS */

    void libxsmm_tpp_convert_at_tensor_to_raw_helper(at::Tensor src_tensor, void **dst_raw_ptr_pt);
#endif

#if defined(NEW_BATCHNORM) || defined(NEW_GROUPNORM) || defined(NEW_CONV)
    /* only for the batchnorm/groupnorm and conv */
    #define CHANNEL_BLOCK_SIZE (64) /* hardcoded for now, used in setup_new() */
#endif

#ifdef NEW_BATCHNORM
    //#include "batchnorm_tpp.h"

    //#define USE_OLD_HANDLE_BN

    //#define DUMP_FORWARD_BN
    //#define DUMP_BACKWARD_BN

#endif /* for NEW_BATCHNORM */

#ifdef NEW_GROUPNORM
    //#include "groupnorm_tpp.h"

    //#define USE_OLD_HANDLE_GN

    //#define DUMP_FORWARD_GN
    //#define DUMP_BACKWARD_GN

#endif /* for NEW_GROUPNORM */

#ifdef NEW_CONV
    /* FIXME */
    #define K_BLOCK_SIZE 64

    //#include "conv_tpp.h"

    //#define USE_OLD_HANDLE_CONV

    //#define DUMP_FORWARD_CONV
    //#define DUMP_BACKWARD_CONV

#endif /* for NEW_CONV */

#ifdef NEW_FC
    /* FIXME */
    #define FC_BLOCK_SIZE 64

    //#include "fullyconnected_tpp.h"

#endif

#ifdef NEW_POOLING
    /* FIXME */
    #define POOLING_BLOCK_SIZE 64

    //#include "pooling_tpp.h"

    //#define USE_OLD_HANDLE_POOLING

#endif

#if defined(NEW_BOTTLENECK) && (!defined(NEW_BATCHNORM) || !defined(NEW_GROUPNORM) || !defined(NEW_CONV))
    #error "For NEW_BOTTLENECK one must also define NEW_BATCHNORM, NEW_GROUPNORM and NEW_CONV"
#endif

const at::ScalarType dt_map[] = {at::kDouble, at::kFloat, at::kBFloat16, at::kLong, at::kInt, at::kShort, at::kChar, at::kByte/*"UNK"*/};

#ifndef PART_OF_EXTENSIONS
void init_libxsmm()
{
  libxsmm_init();
}

#endif

#ifdef OLD_LIBXSMM_HANDLES

void print_layout(std::string desc, libxsmm_dnn_tensor_datalayout *layout, at::Tensor pt_tensor) {
  char *dim_name[] = {"N", "H", "W", "C", "K", "R", "S", "X", "RLM", "RLK", "RLN"};
  char *xsmm_dtypes[] = {"F64", "F32", "BF16", "I32", "I16", "I8", "UNK"};
  int i;
  //return;
  auto ndims = layout->num_dims;
  bool check = true;
  check = check && pt_tensor.dim() == layout->num_dims;
  for(i = 0; i < ndims; i++) {
    check = check && pt_tensor.size(i) == layout->dim_size[ndims - i - 1];
  }
  check = check && pt_tensor.scalar_type() == dt_map[layout->datatype];
  check = check && pt_tensor.is_contiguous();

  if(!check) {
    std::stringstream ss;
    ss << desc << ": F:" << layout->format << " TT: " << layout->tensor_type << " DT: " << xsmm_dtypes[layout->datatype] << " [";
    //printf("%s: F:%d TT: %d DT: %d [", desc, layout->format, layout->tensor_type, layout->datatype);
    for(i = layout->num_dims - 1; i >= 0; i--) {
      //printf("%s:%d%s", dim_name[layout->dim_type[i]], layout->dim_size[i], i == 0 ? "" : ", ");
      ss << dim_name[layout->dim_type[i]] << ":" << layout->dim_size[i] << (i == 0 ? "" : ", ");
    }
    //printf("]\n");
    ss << "]\n";
    ss << "  PyTorch Tensor type: " << pt_tensor.scalar_type() << " size: " << pt_tensor.sizes() << " cont: " << pt_tensor.is_contiguous() << std::endl;
    std::cout << ss.str();
  }
  if(!check) {
    //exit(1);
  }

}

void libxsmm_dnn_conv_set_ptr_helper(libxsmm_dnn_layer *handle, const libxsmm_dnn_tensor_type type, at::Tensor pt_tensor, std::string desc)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  void *ptr;
  if(pt_tensor.scalar_type() == at::kFloat) ptr = (void*)pt_tensor.data_ptr<float>();
  else if(pt_tensor.scalar_type() == at::kBFloat16) ptr = (void*)pt_tensor.data_ptr<at::BFloat16>();
  else if(pt_tensor.scalar_type() == at::kByte) ptr = (void*)pt_tensor.data_ptr();
  else ptr = NULL;
  uintptr_t addr = (uintptr_t)ptr;
  LIBXSMM_ASSERT(ptr != NULL && addr % 64 == 0);
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  if(!tensor) {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_create_tensor_datalayout(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
    PRINT_LAYOUT(desc, layout, pt_tensor);
    tensor = libxsmm_dnn_link_tensor( layout, ptr, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( handle, tensor, type ) );
  } else {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(tensor, ptr) );
  }
}

void libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_dnn_fusedgroupnorm *handle, const libxsmm_dnn_tensor_type type, at::Tensor pt_tensor, std::string desc)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  void *ptr;
  if(pt_tensor.scalar_type() == at::kFloat) ptr = (void*)pt_tensor.data_ptr<float>();
  else if(pt_tensor.scalar_type() == at::kBFloat16) ptr = (void*)pt_tensor.data_ptr<at::BFloat16>();
  else if(pt_tensor.scalar_type() == at::kByte) ptr = (void*)pt_tensor.data_ptr();
  else ptr = NULL;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_fusedgroupnorm_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  if(!tensor) {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_fusedgroupnorm_create_tensor_datalayout(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
    //PRINT_LAYOUT(desc, layout, pt_tensor);
    tensor = libxsmm_dnn_link_tensor( layout, ptr, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedgroupnorm_bind_tensor( handle, tensor, type ) );
  } else {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(tensor, ptr) );
  }
}

void libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_dnn_fusedbatchnorm *handle, const libxsmm_dnn_tensor_type type, at::Tensor pt_tensor, std::string desc)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  void *ptr;
  if(pt_tensor.scalar_type() == at::kFloat) ptr = (void*)pt_tensor.data_ptr<float>();
  else if(pt_tensor.scalar_type() == at::kBFloat16) ptr = (void*)pt_tensor.data_ptr<at::BFloat16>();
  else if(pt_tensor.scalar_type() == at::kByte) ptr = (void*)pt_tensor.data_ptr();
  else ptr = NULL;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_fusedbatchnorm_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  if(!tensor) {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
    //PRINT_LAYOUT(desc, layout, pt_tensor);
    tensor = libxsmm_dnn_link_tensor( layout, ptr, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( handle, tensor, type ) );
  } else {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(tensor, ptr) );
  }
}

void libxsmm_dnn_avg_pooling_set_ptr_helper(libxsmm_dnn_pooling *handle, const libxsmm_dnn_tensor_type type, at::Tensor pt_tensor, std::string desc)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  void *ptr;
  if(pt_tensor.scalar_type() == at::kFloat) ptr = (void*)pt_tensor.data_ptr<float>();
  else if(pt_tensor.scalar_type() == at::kBFloat16) ptr = (void*)pt_tensor.data_ptr<at::BFloat16>();
  else ptr = NULL;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_pooling_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  if(!tensor) {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_pooling_create_tensor_datalayout(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
    //PRINT_LAYOUT(desc, layout, pt_tensor);
    tensor = libxsmm_dnn_link_tensor( layout, ptr, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor( handle, tensor, type ) );
  } else {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(tensor, ptr) );
  }
}

void libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_dnn_pooling *handle, const libxsmm_dnn_tensor_type type, at::Tensor pt_tensor, std::string desc)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  void *ptr;
  if(pt_tensor.scalar_type() == at::kFloat) ptr = (void*)pt_tensor.data_ptr<float>();
  else if(pt_tensor.scalar_type() == at::kBFloat16) ptr = (void*)pt_tensor.data_ptr<at::BFloat16>();
  else if(pt_tensor.scalar_type() == at::kInt) ptr = (void*)pt_tensor.data_ptr<int>();
  else if(pt_tensor.scalar_type() == at::kByte) ptr = (void*)pt_tensor.data_ptr();
  else ptr = NULL;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_pooling_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  if(!tensor) {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_pooling_create_tensor_datalayout(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
    PRINT_LAYOUT(desc, layout, pt_tensor);
    tensor = libxsmm_dnn_link_tensor( layout, ptr, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor( handle, tensor, type ) );
  } else {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(tensor, ptr) );
  }
}
void libxsmm_dnn_convolution_release_tensor_helper(libxsmm_dnn_layer *handle, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  if(tensor) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor(tensor) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( handle, type ) );
  }
}

void libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_dnn_fusedgroupnorm *handle, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_fusedgroupnorm_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  if(tensor) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor(tensor) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedgroupnorm_release_tensor( handle, type ) );
  }
}

void libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_dnn_fusedbatchnorm *handle, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_fusedbatchnorm_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  if(tensor) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor(tensor) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( handle, type ) );
  }
}

void libxsmm_dnn_avg_pooling_release_tensor_helper(libxsmm_dnn_pooling *handle, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_pooling_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  if(tensor) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor(tensor) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_release_tensor( handle, type ) );
  }
}

void libxsmm_dnn_max_pooling_release_tensor_helper(libxsmm_dnn_pooling *handle, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_pooling_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  if(tensor) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor(tensor) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_release_tensor( handle, type ) );
  }
}

void *conv_create_handle(int N, int C, int H, int W, int K, int R, int S, int padding, int stride, int dtype)
{
  libxsmm_dnn_conv_desc conv_desc;
  libxsmm_dnn_layer* libxsmm_handle;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  conv_desc.N = N;
  conv_desc.C = C;
  conv_desc.H = H;
  conv_desc.W = W;
  conv_desc.K = K;
  conv_desc.R = R;
  conv_desc.S = S;
  conv_desc.u = stride;
  conv_desc.v = stride;
  conv_desc.pad_h_in = 0;
  conv_desc.pad_w_in = 0;

  conv_desc.pad_w = padding;
  conv_desc.pad_h = padding;

  conv_desc.pad_h_out = 0;
  conv_desc.pad_w_out = 0;

  conv_desc.threads = omp_get_max_threads();
  conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
  conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
  conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;

  if(dtype == 2)
  {
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_BF16;
  }
  else if(dtype == 1)
  {
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  }

  libxsmm_handle = libxsmm_dnn_create_conv_layer(conv_desc, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto s_size = libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto scratch = _mm_malloc( s_size, 2097152 );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
  return (void *)libxsmm_handle;
}

void *fusedgroupnorm_create_handle(int N, int C, int H, int W, int G, int dtype, int relu, int eltwise)
{
  libxsmm_dnn_fusedgroupnorm_desc fusedgroupnorm_desc;
  libxsmm_dnn_fusedgroupnorm* libxsmm_handle;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  fusedgroupnorm_desc.N = N;
  fusedgroupnorm_desc.C = C;
  fusedgroupnorm_desc.G = G;
  fusedgroupnorm_desc.H = H;
  fusedgroupnorm_desc.W = W;
  fusedgroupnorm_desc.u = 1;
  fusedgroupnorm_desc.v = 1;
  fusedgroupnorm_desc.pad_h_in = 0;
  fusedgroupnorm_desc.pad_w_in = 0;
  fusedgroupnorm_desc.pad_h_out = 0;
  fusedgroupnorm_desc.pad_w_out = 0;
  fusedgroupnorm_desc.threads = omp_get_max_threads();
  fusedgroupnorm_desc.datatype_in = (dtype == 1 ? LIBXSMM_DNN_DATATYPE_F32 : LIBXSMM_DNN_DATATYPE_BF16);
  fusedgroupnorm_desc.datatype_out = (dtype == 1 ? LIBXSMM_DNN_DATATYPE_F32 : LIBXSMM_DNN_DATATYPE_BF16);
  fusedgroupnorm_desc.datatype_stats = LIBXSMM_DNN_DATATYPE_F32;
  fusedgroupnorm_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fusedgroupnorm_desc.fuse_order = LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU;
  fusedgroupnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDGN_OPS_GN;

  if(relu)
    fusedgroupnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK;

  if(eltwise)
    fusedgroupnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE;

  if(relu & eltwise)
    fusedgroupnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK;

  libxsmm_handle = libxsmm_dnn_create_fusedgroupnorm( fusedgroupnorm_desc, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto s_size = libxsmm_dnn_fusedgroupnorm_get_scratch_size( libxsmm_handle, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto  scratch = _mm_malloc( s_size, 2097152 );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedgroupnorm_bind_scratch( libxsmm_handle, scratch ) );
  //std::cout << "Create Handle = " << libxsmm_handle << std::endl;
  return (void *)libxsmm_handle;
}

void *fusedbatchnorm_create_handle(int N, int C, int H, int W, int dtype, bool relu, bool eltwise, bool train)
{
  libxsmm_dnn_fusedbatchnorm_desc fusedbatchnorm_desc;
  libxsmm_dnn_fusedbatchnorm* libxsmm_handle;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  fusedbatchnorm_desc.fullN = N;
  fusedbatchnorm_desc.partN = N;
  fusedbatchnorm_desc.C = C;
  fusedbatchnorm_desc.H = H;
  fusedbatchnorm_desc.W = W;
  fusedbatchnorm_desc.u = 1;
  fusedbatchnorm_desc.v = 1;
  fusedbatchnorm_desc.pad_h_in = 0;
  fusedbatchnorm_desc.pad_w_in = 0;
  fusedbatchnorm_desc.pad_h_out = 0;
  fusedbatchnorm_desc.pad_w_out = 0;
  fusedbatchnorm_desc.threads = omp_get_max_threads();
  fusedbatchnorm_desc.datatype_in = (dtype == 1 ? LIBXSMM_DNN_DATATYPE_F32 : LIBXSMM_DNN_DATATYPE_BF16);
  fusedbatchnorm_desc.datatype_out = (dtype == 1 ? LIBXSMM_DNN_DATATYPE_F32 : LIBXSMM_DNN_DATATYPE_BF16);
  fusedbatchnorm_desc.datatype_stats = LIBXSMM_DNN_DATATYPE_F32;
  fusedbatchnorm_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fusedbatchnorm_desc.fuse_order = LIBXSMM_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU;
  fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN;

  if(relu & train)
    fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_RELU_WITH_MASK;
  else if (relu & !train)
    fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_RELU_WITH_MASK;

  if(eltwise & train)
    fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE;
  else if(eltwise & !train)
    fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE;

  if(relu & eltwise & train)
    fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE_RELU_WITH_MASK;
  else if(relu & eltwise & !train)
    fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU_WITH_MASK;

  libxsmm_handle = libxsmm_dnn_create_fusedbatchnorm( fusedbatchnorm_desc, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto s_size = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto  scratch = _mm_malloc( s_size, 2097152 );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle, scratch ) );
  //std::cout << "Create Handle = " << libxsmm_handle << std::endl;
  return (void *)libxsmm_handle;
}

void *avg_pooling_create_handle(int N, int C, int H, int W, int R, int S, int padding, int stride, int dtype)
{
  libxsmm_dnn_pooling_desc pooling_desc;
  libxsmm_dnn_pooling* libxsmm_handle;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;

  pooling_desc.N = N;
  pooling_desc.C = C;
  pooling_desc.H = H;
  pooling_desc.W = W;
  pooling_desc.u = stride;
  pooling_desc.v = stride;
  pooling_desc.R = R;
  pooling_desc.S = S;
  pooling_desc.pad_h = padding;
  pooling_desc.pad_w = padding;
  pooling_desc.pad_h_in = 0;
  pooling_desc.pad_w_in = 0;
  pooling_desc.pad_h_out = 0;
  pooling_desc.pad_w_out = 0;
  pooling_desc.threads = omp_get_max_threads();
  pooling_desc.datatype_in = (dtype == 1 ? LIBXSMM_DNN_DATATYPE_F32 : LIBXSMM_DNN_DATATYPE_BF16);
  pooling_desc.datatype_out = (dtype == 1 ? LIBXSMM_DNN_DATATYPE_F32 : LIBXSMM_DNN_DATATYPE_BF16);
  pooling_desc.datatype_mask = LIBXSMM_DNN_DATATYPE_I32;
  pooling_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  pooling_desc.pooling_type = LIBXSMM_DNN_POOLING_AVG;

  libxsmm_handle = libxsmm_dnn_create_pooling( pooling_desc, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto s_size = libxsmm_dnn_pooling_get_scratch_size( libxsmm_handle, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto  scratch = _mm_malloc( s_size, 2097152 );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_scratch( libxsmm_handle, scratch ) );
  return (void*)libxsmm_handle;
}

void *max_pooling_create_handle(int N, int C, int H, int W, int R, int S, int padding, int stride, int dtype)
{
  libxsmm_dnn_pooling_desc pooling_desc;
  libxsmm_dnn_pooling* libxsmm_handle;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;

  pooling_desc.N = N;
  pooling_desc.C = C;
  pooling_desc.H = H;
  pooling_desc.W = W;
  pooling_desc.u = stride;
  pooling_desc.v = stride;
  pooling_desc.R = R;
  pooling_desc.S = S;
  pooling_desc.pad_h = padding;
  pooling_desc.pad_w = padding;
  pooling_desc.pad_h_in = 0;
  pooling_desc.pad_w_in = 0;
  pooling_desc.pad_h_out = 0;
  pooling_desc.pad_w_out = 0;
  pooling_desc.threads = omp_get_max_threads();
  pooling_desc.datatype_in = (dtype == 1 ? LIBXSMM_DNN_DATATYPE_F32 : LIBXSMM_DNN_DATATYPE_BF16);
  pooling_desc.datatype_out = (dtype == 1 ? LIBXSMM_DNN_DATATYPE_F32 : LIBXSMM_DNN_DATATYPE_BF16);
  pooling_desc.datatype_mask = LIBXSMM_DNN_DATATYPE_I32;
  pooling_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  pooling_desc.pooling_type = LIBXSMM_DNN_POOLING_MAX;

  libxsmm_handle = libxsmm_dnn_create_pooling( pooling_desc, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto s_size = libxsmm_dnn_pooling_get_scratch_size( libxsmm_handle, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto  scratch = _mm_malloc( s_size, 2097152 );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_scratch( libxsmm_handle, scratch ) );
  return (void*)libxsmm_handle;
}

std::vector<long> get_conv_tensor_layout(void *libxsmm_handle_, std::string tensor_type)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_layer* libxsmm_handle = (libxsmm_dnn_layer*)libxsmm_handle_;
  std::vector<long> dim_size;

  if(tensor_type == "output")
  {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status); 
    CHKERR_LIBXSMM_DNN( status );

    for(int i = layout->num_dims - 1; i >= 0; i--) {
      dim_size.push_back(layout->dim_size[i]);
    }
    libxsmm_dnn_destroy_tensor_datalayout( layout );
  }
  else if(tensor_type == "grad_input")
  {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status); 
    CHKERR_LIBXSMM_DNN( status );

    for(int i = layout->num_dims - 1; i >= 0; i--) {
      dim_size.push_back(layout->dim_size[i]);
    }
    libxsmm_dnn_destroy_tensor_datalayout( layout );
  }
  else if(tensor_type == "weight")
  {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, &status); 
    CHKERR_LIBXSMM_DNN( status );

    for(int i = layout->num_dims - 1; i >= 0; i--) {
      dim_size.push_back(layout->dim_size[i]);
    }
    libxsmm_dnn_destroy_tensor_datalayout( layout );
  }
  else if(tensor_type == "grad_weight")
  {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER, &status); 
    CHKERR_LIBXSMM_DNN( status );

    for(int i = layout->num_dims - 1; i >= 0; i--) {
      dim_size.push_back(layout->dim_size[i]);
    }
    libxsmm_dnn_destroy_tensor_datalayout( layout );
  }
  return dim_size;
}

at::Tensor conv_forward(void *libxsmm_handle_, torch::Tensor input, torch::Tensor weight, std::vector<long> dim_size)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_layer* libxsmm_handle = (libxsmm_dnn_layer*)libxsmm_handle_;

  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, weight, "Weight");

  at::Tensor output = input.new_empty(dim_size);
  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, output, "Output");

  {
    RECORD_FUNCTION("xsmm_conv_fwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
    }
  }
  return output;
}

std::vector<at::Tensor> conv_backward(void *libxsmm_handle_, torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_layer* libxsmm_handle = (libxsmm_dnn_layer*)libxsmm_handle_;

  auto grad_input  = at::empty(input.sizes(), input.options());
  auto grad_weight = at::empty(weight.sizes(), weight.options());

  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, grad_output, "Grad Output");
  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, weight, "Weight");
  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, grad_input, "Grad Input");
  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER, grad_weight, "Grad Weight");
  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");

  if(input.requires_grad())
  {
    RECORD_FUNCTION("xsmm_conv_bwdupd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid) );
    }
  }
  else
  {
    RECORD_FUNCTION("xsmm_conv_upd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid) );
    }
  }

  return {grad_input, grad_weight};
}

std::vector<long> get_gn_tensor_layout(void *libxsmm_handle_, std::string tensor_type)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_fusedgroupnorm* libxsmm_handle = (libxsmm_dnn_fusedgroupnorm*)libxsmm_handle_;
  std::vector<long> dim_size;

  if(tensor_type == "mean")
  {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_fusedgroupnorm_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status); 
    CHKERR_LIBXSMM_DNN( status );
    for(int i = layout->num_dims - 1; i >= 0; i--) {
      dim_size.push_back(layout->dim_size[i]);
    }
    libxsmm_dnn_destroy_tensor_datalayout( layout );
  }
  else if(tensor_type == "output")
  {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_fusedgroupnorm_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status); 
    CHKERR_LIBXSMM_DNN( status );
    for(int i = layout->num_dims - 1; i >= 0; i--) {
      dim_size.push_back(layout->dim_size[i]);
    }
    libxsmm_dnn_destroy_tensor_datalayout( layout );
  }
  return dim_size;
}

std::vector<at::Tensor> gnorm_forward(void *libxsmm_handle_, torch::Tensor input, torch::Tensor input_add, torch::Tensor gamma, torch::Tensor beta, 
                                      std::vector<long> stats_size, std::vector<long> output_size)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_fusedgroupnorm* libxsmm_handle = (libxsmm_dnn_fusedgroupnorm*)libxsmm_handle_;
  
  at::Tensor mean = at::empty(stats_size, torch::TensorOptions().dtype(dt_map[1])); 
  at::Tensor var = at::empty(stats_size, torch::TensorOptions().dtype(dt_map[1])); 
  at::Tensor invstd = at::empty(stats_size, torch::TensorOptions().dtype(dt_map[1]));
  at::Tensor relu_mask = at::empty(output_size, torch::TensorOptions().dtype(dt_map[7]));
  at::Tensor output;
  output = input.new_empty(output_size);

  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  if(input_add.numel() != 0)
    libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ADD, input_add, "Residual Input");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, gamma, "Gamma");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BETA, beta, "Beta");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL, mean, "Mean");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_VARIANCE, var, "Variance");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_RCPSTDDEV, invstd, "Inv Stddev");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_RELU_MASK, relu_mask, "ReLU Mask");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, output, "Output");

  {
    RECORD_FUNCTION("xsmm_gn_fwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedgroupnorm_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
    }
  }

  return {output, mean, var, invstd, relu_mask};
}

std::vector<at::Tensor> gnorm_backward(void *libxsmm_handle_, torch::Tensor grad_output, torch::Tensor input, torch::Tensor input_add, 
                                        torch::Tensor gamma, torch::Tensor output, torch::Tensor mean, torch::Tensor var, 
                                        torch::Tensor invstd, torch::Tensor relu_mask)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;

  auto grad_input = at::empty(input.sizes(), input.options());

  at::Tensor grad_input_add;
  if(input_add.numel() != 0)
    grad_input_add = at::empty(input_add.sizes(), input_add.options());

  auto grad_gamma = at::empty(gamma.sizes(), gamma.options());
  auto grad_beta = at::empty(gamma.sizes(), gamma.options());

  libxsmm_dnn_fusedgroupnorm* libxsmm_handle = (libxsmm_dnn_fusedgroupnorm*)libxsmm_handle_;
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, gamma, "Gamma");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL, mean, "Mean");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_VARIANCE, var, "Variance");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_RCPSTDDEV, invstd, "Inverse Std");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_RELU_MASK, relu_mask, "ReLU Mask");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, grad_gamma, "GradGamma");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, grad_beta, "GradBeta");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, grad_output, "GradOutput");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, grad_input, "GradInput");

  if(input_add.numel() != 0)
    if(grad_input_add.numel() != 0)
      libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_ADD, grad_input_add, "GradInputAdd");
  {
    RECORD_FUNCTION("xsmm_gn_bwd", std::vector<c10::IValue>(/*grad_output, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedgroupnorm_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid) );
    }
  }

  return {grad_input, grad_input_add, grad_gamma, grad_beta};
}

std::vector<long> get_bn_tensor_layout(void *libxsmm_handle_, std::string tensor_type)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_fusedbatchnorm* libxsmm_handle = (libxsmm_dnn_fusedbatchnorm*)libxsmm_handle_;
  std::vector<long> dim_size;

  if(tensor_type == "output")
  {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status); 
    CHKERR_LIBXSMM_DNN( status );
    for(int i = layout->num_dims - 1; i >= 0; i--) {
      dim_size.push_back(layout->dim_size[i]);
    }
    libxsmm_dnn_destroy_tensor_datalayout( layout );
  }
  return dim_size;
}


std::vector<at::Tensor> bnorm_forward(void *libxsmm_handle_, torch::Tensor input, torch::Tensor input_add, torch::Tensor gamma, torch::Tensor beta, 
                                        torch::Tensor mean, torch::Tensor var, torch::Tensor invstd, std::vector<long> output_size)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_fusedbatchnorm* libxsmm_handle = (libxsmm_dnn_fusedbatchnorm*)libxsmm_handle_;
  
  at::Tensor relu_mask = at::empty(output_size, torch::TensorOptions().dtype(dt_map[7]));
  at::Tensor output;
  output = input.new_empty(output_size);

  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  if(input_add.numel() > 0)
    libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ADD, input_add, "Residual Input");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, gamma, "Gamma");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BETA, beta, "Beta");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL, mean, "Mean");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_VARIANCE, var, "Variance");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_RCPSTDDEV, invstd, "Inv Stddev");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_RELU_MASK, relu_mask, "ReLU Mask");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, output, "Output");

  {
    RECORD_FUNCTION("xsmm_bn_fwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
    }
  }

  return {output, relu_mask};
}


std::vector<at::Tensor> bnorm_backward(void *libxsmm_handle_, torch::Tensor grad_output, torch::Tensor input, torch::Tensor input_add, 
                                        torch::Tensor gamma, torch::Tensor output, torch::Tensor mean, torch::Tensor var, 
                                        torch::Tensor invstd, torch::Tensor relu_mask)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;

  auto grad_input = input.new_empty(input.sizes());

  at::Tensor grad_input_add;
  if(input_add.numel() > 0)
    grad_input_add = input_add.new_empty(input_add.sizes());

  auto grad_gamma = gamma.new_empty(gamma.sizes());
  auto grad_beta = gamma.new_empty(gamma.sizes());

  libxsmm_dnn_fusedbatchnorm* libxsmm_handle = (libxsmm_dnn_fusedbatchnorm*)libxsmm_handle_;
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, gamma, "Gamma");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL, mean, "Mean");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_VARIANCE, var, "Variance");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_RCPSTDDEV, invstd, "Inverse Std");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_RELU_MASK, relu_mask, "ReLU Mask");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, grad_gamma, "GradGamma");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, grad_beta, "GradBeta");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, grad_output, "GradOutput");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, grad_input, "GradInput");

  if(grad_input_add.numel() > 0)
    libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_ADD, grad_input_add, "GradInputAdd");
  {
    RECORD_FUNCTION("xsmm_bn_bwd", std::vector<c10::IValue>(/*grad_output, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid) );
    }
  }

  return {grad_input, grad_input_add, grad_gamma, grad_beta};
}


std::vector<long> get_pooling_tensor_layout(void *libxsmm_handle_, std::string tensor_type)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_pooling* libxsmm_handle = (libxsmm_dnn_pooling*)libxsmm_handle_;
  std::vector<long> dim_size;

  if(tensor_type == "output")
  {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_pooling_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status); 
    CHKERR_LIBXSMM_DNN( status );
    for(int i = layout->num_dims - 1; i >= 0; i--) {
      dim_size.push_back(layout->dim_size[i]);
    }
    libxsmm_dnn_destroy_tensor_datalayout( layout );
  }
  else if(tensor_type == "grad_input")
  {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_pooling_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status); 
    CHKERR_LIBXSMM_DNN( status );
    for(int i = layout->num_dims - 1; i >= 0; i--) {
      dim_size.push_back(layout->dim_size[i]);
    }
    libxsmm_dnn_destroy_tensor_datalayout( layout );
  }
  return dim_size;
}

at::Tensor avg_pooling_forward(void *libxsmm_handle_, torch::Tensor input, std::vector<long> output_size)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_pooling* libxsmm_handle = (libxsmm_dnn_pooling*)libxsmm_handle_;

  libxsmm_dnn_avg_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");

  at::Tensor output;
  output = input.new_empty(output_size);
  libxsmm_dnn_avg_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, output, "Output");

  {
    RECORD_FUNCTION("xsmm_pooling_fwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
    }
  }
  return output;
}

at::Tensor avg_pooling_backward(void *libxsmm_handle_, torch::Tensor grad_output, std::vector<long> grad_in_size)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_pooling* libxsmm_handle = (libxsmm_dnn_pooling*)libxsmm_handle_;

  libxsmm_dnn_avg_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, grad_output, "Grad Output");

  at::Tensor grad_input = at::empty(grad_in_size, grad_output.options());
  libxsmm_dnn_avg_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, grad_input, "Grad Input");

  {
    RECORD_FUNCTION("xsmm_pooling_bwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid) );
    }
  }
  return grad_input;
}

std::vector<at::Tensor> max_pooling_forward(void *libxsmm_handle_, torch::Tensor input, std::vector<long> output_size)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_pooling* libxsmm_handle = (libxsmm_dnn_pooling*)libxsmm_handle_;

  libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");

  at::Tensor pool_mask = at::empty(output_size, torch::TensorOptions().dtype(at::kInt));
  libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_POOLING_MASK, pool_mask, "Pooling Mask");

  at::Tensor output = input.new_empty(output_size);

  libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, output, "Output");

  {
    RECORD_FUNCTION("xsmm_pooling_fwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
    }
  }
  return {output, pool_mask};
}

at::Tensor max_pooling_backward(void *libxsmm_handle_, torch::Tensor grad_output, torch::Tensor pool_mask, std::vector<long> grad_in_size)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_pooling* libxsmm_handle = (libxsmm_dnn_pooling*)libxsmm_handle_;

  libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, grad_output, "Grad Output");
  libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_POOLING_MASK, pool_mask, "Pooling Mask");

  at::Tensor grad_input = at::empty(grad_in_size, grad_output.options());
  libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, grad_input, "Grad Input");

  {
    RECORD_FUNCTION("xsmm_pooling_bwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid) );
    }
  }
  return grad_input;
}

void fusedgroupnorm_destroy_handle( void* libxsmm_handle_ )
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_fusedgroupnorm* libxsmm_handle = (libxsmm_dnn_fusedgroupnorm*)libxsmm_handle_;
  //std::cout << "Destroy Handle = " << libxsmm_handle << std::endl;

  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_RELU_MASK);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ADD);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BETA);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_VARIANCE);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_RCPSTDDEV);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_ADD);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA);
  libxsmm_dnn_fusedgroupnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT);
  size_t scratch_size = libxsmm_dnn_fusedgroupnorm_get_scratch_size( libxsmm_handle, &status );
  if(scratch_size > 0) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedgroupnorm_release_scratch( libxsmm_handle ) );
  }

  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_fusedgroupnorm( libxsmm_handle ) );
}

void conv_destroy_handle( void* libxsmm_handle_ )
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_layer* libxsmm_handle = (libxsmm_dnn_layer*)libxsmm_handle_;
  //std::cout << "Destroy Handle = " << libxsmm_handle << std::endl;

  libxsmm_dnn_convolution_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT);
  libxsmm_dnn_convolution_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER);
  libxsmm_dnn_convolution_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT);
  libxsmm_dnn_convolution_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT);
  libxsmm_dnn_convolution_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER);
  libxsmm_dnn_convolution_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT);
  size_t scratch_size = libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
  if(scratch_size > 0) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL) );
  }

  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_conv_layer( libxsmm_handle ) );
}

void fusedbatchnorm_destroy_handle( void* libxsmm_handle_ )
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_fusedbatchnorm* libxsmm_handle = (libxsmm_dnn_fusedbatchnorm*)libxsmm_handle_;
  //std::cout << "Destroy Handle = " << libxsmm_handle << std::endl;

  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_RELU_MASK);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ADD);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BETA);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_VARIANCE);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_RCPSTDDEV);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_ADD);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA);
  libxsmm_dnn_fusedbatchnorm_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT);
  size_t scratch_size = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle, &status );
  if(scratch_size > 0) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_scratch( libxsmm_handle ) );
  }

  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_fusedbatchnorm( libxsmm_handle ) );
}

void avg_pooling_destroy_handle( void* libxsmm_handle_ )
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_pooling* libxsmm_handle = (libxsmm_dnn_pooling*)libxsmm_handle_;
  //std::cout << "Destroy Handle = " << libxsmm_handle << std::endl;

  libxsmm_dnn_avg_pooling_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT);
  libxsmm_dnn_avg_pooling_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT);
  libxsmm_dnn_avg_pooling_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT);
  libxsmm_dnn_avg_pooling_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT);
  size_t scratch_size = libxsmm_dnn_pooling_get_scratch_size( libxsmm_handle, &status );
  if(scratch_size > 0) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_release_scratch(libxsmm_handle) );
  }

  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_pooling( libxsmm_handle ) );
}

void max_pooling_destroy_handle( void* libxsmm_handle_ )
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_pooling* libxsmm_handle = (libxsmm_dnn_pooling*)libxsmm_handle_;
  //std::cout << "Destroy Handle = " << libxsmm_handle << std::endl;

  libxsmm_dnn_max_pooling_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT);
  libxsmm_dnn_max_pooling_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_POOLING_MASK);
  libxsmm_dnn_max_pooling_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT);
  libxsmm_dnn_max_pooling_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT);
  libxsmm_dnn_max_pooling_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT);
  size_t scratch_size = libxsmm_dnn_pooling_get_scratch_size( libxsmm_handle, &status );
  if(scratch_size > 0) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_release_scratch(libxsmm_handle) );
  }

  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_pooling( libxsmm_handle ) );
}

#endif /* OLD_LIBXSMM_HANDLES */


#if defined(NEW_BATCHNORM) || defined(NEW_GROUPNORM) || defined(NEW_CONV) || defined(NEW_FC) || defined(NEW_POOLING)

void libxsmm_tpp_convert_at_tensor_to_raw_helper(at::Tensor src_tensor, void **dst_raw_ptr_pt)
{
  void *ptr;
  if(src_tensor.scalar_type() == at::kDouble)        ptr = (void*)src_tensor.data_ptr<double>();
  else if(src_tensor.scalar_type() == at::kFloat)    ptr = (void*)src_tensor.data_ptr<float>();
  else if(src_tensor.scalar_type() == at::kInt)      ptr = (void*)src_tensor.data_ptr<int>();
  else if(src_tensor.scalar_type() == at::kBFloat16) ptr = (void*)src_tensor.data_ptr<at::BFloat16>();
  else if(src_tensor.scalar_type() == at::kByte)     ptr = (void*)src_tensor.data_ptr();
  else {
    std::cout << "Error: unrecognized scalar_type " <<  src_tensor.scalar_type() << " for the at::Tensor in libxsmm_tpp_convert_at_tensor_to_raw_helper\n";
    ptr = NULL;
  }

  *dst_raw_ptr_pt = ptr;
}

#endif

#ifdef NEW_BATCHNORM

typedef struct pcl_cgbp_bn_config {
  libxsmm_dnn_bn_fwd_config fwd_cfg;
  libxsmm_dnn_bn_bwd_config bwd_cfg;
  int              dtype;               /* 0 for fp32, 1 for bf16 */
  void*            scratch;
//  my_normalization_norm_type norm_type;
  float            eps;
#ifdef USE_OLD_HANDLE_BN
  void*            libxsmm_handle_;
#endif
} pcl_cgbp_bn_config;

int bnorm_get_c_block( int C /*, datatype as an int flag? */ )
{
  libxsmm_blasint bc = CHANNEL_BLOCK_SIZE; /* hardcoded for now */

  if (C % bc != 0)
    bc = C;

  return bc;
}

pcl_cgbp_bn_config bnorm_setup_new( libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W,
                              libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in, libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                              float eps, int fuse_type_int, int dtype_int, int bc_or_negative ) {
  pcl_cgbp_bn_config res;

  libxsmm_dnn_bn_fuse fuse_type_enum = LIBXSMM_DNN_BN_FUSE_NONE;

  switch(fuse_type_int) {
    case 0:
      fuse_type_enum = LIBXSMM_DNN_BN_FUSE_NONE;
      break;
    case 1:
      fuse_type_enum = LIBXSMM_DNN_BN_FUSE_RELU;
      break;
    case 2:
      fuse_type_enum = LIBXSMM_DNN_BN_FUSE_ELTWISE;
      break;
    case 3:
      fuse_type_enum = LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU;
      break;
    case 4:
      fuse_type_enum = LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK;
      break;
    case 5:
      fuse_type_enum = LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK;
      break;
    default:
      printf("Unrecognized fuse_type (int) = %d in bnorm_setup_new (assuming FUSE_NONE)\n", fuse_type_int);
      break;
  }

  libxsmm_blasint bc;
  if (bc_or_negative < 0)
    bc = bnorm_get_c_block(C);
  else
    bc = bc_or_negative;
  libxsmm_blasint threads = (libxsmm_blasint)omp_get_max_threads();

  libxsmm_datatype bn_dtype_in  = (dtype_int == 0 ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16);
  libxsmm_datatype bn_dtype_out = bn_dtype_in;
  libxsmm_datatype bn_dtype_comp = LIBXSMM_DATATYPE_F32;
  res.dtype = dtype_int;

  res.fwd_cfg = setup_libxsmm_dnn_bn_fwd(N, C, H, W, bc, pad_h_in, pad_w_in, pad_h_out, pad_w_out, threads, fuse_type_enum, bn_dtype_in, bn_dtype_out, bn_dtype_comp);
  res.bwd_cfg = setup_libxsmm_dnn_bn_bwd(N, C, H, W, bc, pad_h_in, pad_w_in, pad_h_out, pad_w_out, threads, fuse_type_enum, bn_dtype_in, bn_dtype_out, bn_dtype_comp);

  /* allocate and bind scratch */
  void *scratch = NULL;
  if ( res.fwd_cfg.scratch_size > 0 || res.bwd_cfg.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( res.fwd_cfg.scratch_size, res.bwd_cfg.scratch_size);
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    //init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
    zero_buf((float*)scratch, (alloc_size)/4);
  }

  res.scratch = scratch;
  res.eps     = eps;

#ifdef USE_OLD_HANDLE_BN
  bool relu = false, eltwise = false, train = false;
  if (fuse_type_enum == LIBXSMM_DNN_BN_FUSE_RELU || fuse_type_enum == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || fuse_type_enum == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
    relu = true;
  if (fuse_type_enum == LIBXSMM_DNN_BN_FUSE_ELTWISE || fuse_type_enum == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || fuse_type_enum == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
    eltwise = true;
  if (fuse_type_enum == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || fuse_type_enum == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
    train = true;
  res.libxsmm_handle_ = fusedbatchnorm_create_handle(N, C, H, W, 1 /*int dtype*/, relu, eltwise, train);
#endif

  return res;
}

void bnorm_setup_destroy_new(pcl_cgbp_bn_config cfg)
{
  if (cfg.scratch)
      libxsmm_free(cfg.scratch);
  cfg.scratch = NULL;

  destroy_libxsmm_dnn_bn_fwd(&cfg.fwd_cfg);
  destroy_libxsmm_dnn_bn_bwd(&cfg.bwd_cfg);

#ifdef USE_OLD_HANDLE_BN
  fusedbatchnorm_destroy_handle(cfg.libxsmm_handle_);
  cfg.libxsmm_handle_ = NULL;
#endif

  return;
}

std::vector<at::Tensor> bnorm_forward_new( pcl_cgbp_bn_config& cfg, torch::Tensor input, torch::Tensor input_add, torch::Tensor gamma, torch::Tensor beta,
                                        torch::Tensor mean, torch::Tensor var, torch::Tensor invstd, std::vector<long> output_size, int norm_type)
{
   /* printf("debug: calling bnorm_forward_new with tensor N C H W bc fuse_type norm_type: %d %d %d %d %d %d %d \n", cfg.fwd_cfg.N, cfg.fwd_cfg.C,
              cfg.fwd_cfg.H, cfg.fwd_cfg.W, cfg.fwd_cfg.bc, (int)cfg.fwd_cfg.fuse_type, norm_type); */

#ifdef USE_OLD_HANDLE_BN
  /* printf("debug: in bnorm_forward_new but with an old implementation \n"); */

  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_fusedbatchnorm* libxsmm_handle = (libxsmm_dnn_fusedbatchnorm*)(cfg.libxsmm_handle_);

  at::Tensor relu_mask = at::empty(output_size, torch::TensorOptions().dtype(dt_map[7]));
  at::Tensor output;
  output = input.new_empty(output_size);

  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  if(input_add.numel() > 0)
    libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ADD, input_add, "Residual Input");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, gamma, "Gamma");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BETA, beta, "Beta");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL, mean, "Mean");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_VARIANCE, var, "Variance");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_RCPSTDDEV, invstd, "Inv Stddev");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_RELU_MASK, relu_mask, "ReLU Mask");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, output, "Output");

  {
    RECORD_FUNCTION("xsmm_bn_fwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
    }
  }

#ifdef DUMP_FORWARD_BN
  void *gamma_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(gamma, &gamma_pt);

  void *beta_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(beta, &beta_pt);

  void *mean_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(mean, &mean_pt);

  void *var_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(var, &var_pt);

  void *invstd_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(invstd, &invstd_pt);

  void *input_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(input, &input_pt);

  void *input_add_pt;
  if(input_add.numel() > 0)
    libxsmm_tpp_convert_at_tensor_to_raw_helper(input_add, &input_add_pt);

  void *output_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(output, &output_pt);

  void *relu_mask_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(relu_mask, &relu_mask_pt);

  FILE *f_gamma = fopen("gamma_old.txt", "wt");
  fprintf(f_gamma,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_gamma,"%6.6f\n", ((float*)gamma_pt)[i]);
  fclose(f_gamma);

  FILE *f_beta = fopen("beta_old.txt", "wt");
  fprintf(f_beta,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_beta,"%6.6f\n", ((float*)beta_pt)[i]);
  fclose(f_beta);

  FILE *f_mean = fopen("mean_old.txt", "wt");
  fprintf(f_mean,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_mean,"%6.6f\n", ((float*)mean_pt)[i]);
  fclose(f_mean);

  FILE *f_var = fopen("var_old.txt", "wt");
  fprintf(f_var,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_var,"%6.6f\n", ((float*)var_pt)[i]);
  fclose(f_var);

  FILE *f_invstd = fopen("invstd_old.txt", "wt");
  fprintf(f_invstd,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_invstd,"%6.6f\n", ((float*)invstd_pt)[i]);
  fclose(f_invstd);

  FILE *f_input = fopen("input_old.txt", "wt");
  fprintf(f_input,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_input,"%6.6f\n", ((float*)input_pt)[i]);
  fclose(f_input);

  if(input_add.numel() > 0) {
    FILE *f_input_add = fopen("input_add_old.txt", "wt");
    fprintf(f_input_add,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
    for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
      fprintf(f_input_add,"%6.6f\n", ((float*)input_add_pt)[i]);
    fclose(f_input_add);
  }

  FILE *f_output = fopen("output_old.txt", "wt");
  fprintf(f_output,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_output,"%6.6f\n", ((float*)output_pt)[i]);
  fclose(f_output);

  FILE *f_relu_mask = fopen("relu_mask_old.txt", "wt");
  fprintf(f_relu_mask,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_relu_mask,"%u\n", ((unsigned char*)relu_mask_pt)[i]);
  fclose(f_relu_mask);

  /* exit(-1); */
#endif /* DUMP_FORWARD_BN */

  return {output, relu_mask};

#else


  /* printf("debug: in bnorm_forward_new with new impl\n"); */
/*
  if ( cfg.fwd_cfg.scratch_size > 0 || cfg.bwd_cfg.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( cfg.fwd_cfg.scratch_size, cfg.bwd_cfg.scratch_size);
    init_buf( (float*)(cfg.scratch), (alloc_size)/4, 0, 0 );
  }
*/
  at::Tensor relu_mask = at::empty(output_size, torch::TensorOptions().dtype(dt_map[7]));
  at::Tensor output;
  output = input.new_empty(output_size);

  void *input_pt = NULL, *input_add_pt = NULL, *gamma_pt = NULL, *beta_pt = NULL, *mean_pt = NULL, *var_pt = NULL, *output_pt = NULL, *relu_mask_pt = NULL;

  libxsmm_tpp_convert_at_tensor_to_raw_helper(input,     &input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(gamma,     &gamma_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(beta,      &beta_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(mean,      &mean_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(var,       &var_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(output,    &output_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(relu_mask, &relu_mask_pt);

  if(input_add.numel() > 0)
    libxsmm_tpp_convert_at_tensor_to_raw_helper(input_add, &input_add_pt);

#ifdef RECORD_FUNCTIONS_MACRO
  std::string label{"xsmm_bn_fwd_new"};
  label += "_";
  label += std::to_string(cfg.fwd_cfg.N);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.C);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.H);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.W);
  //std::cout << "label = " << label << "\n";
  //printf("label = %s\n", label);
#endif

  {
    //RECORD_FUNCTION("xsmm_bn_fwd_new", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef RECORD_FUNCTIONS_MACRO
    RECORD_FUNCTION(label, std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      if (cfg.dtype == 0)
        libxsmm_dnn_bn_fwd_exec_f32( cfg.fwd_cfg, (const float*)input_pt, (const float*)input_add_pt, (const float*)gamma_pt, (const float*)beta_pt, (float*)mean_pt, (float*)var_pt,
                            (float*)output_pt, (unsigned char*)relu_mask_pt, cfg.eps, 0, tid, cfg.scratch, (norm_type == 0 ? LIBXSMM_DNN_BN_FULL_NORM : LIBXSMM_DNN_BN_SCALE_ONLY) );
      else
        libxsmm_dnn_bn_fwd_exec_bf16( cfg.fwd_cfg, (const libxsmm_bfloat16*)input_pt, (const libxsmm_bfloat16*)input_add_pt, (const float*)gamma_pt, (const float*)beta_pt, (float*)mean_pt, (float*)var_pt,
                            (libxsmm_bfloat16*)output_pt, (unsigned char*)relu_mask_pt, cfg.eps, 0, tid, cfg.scratch, (norm_type == 0 ? LIBXSMM_DNN_BN_FULL_NORM : LIBXSMM_DNN_BN_SCALE_ONLY) );
    }
  }

#ifdef DUMP_FORWARD_BN
  void *invstd_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(invstd, &invstd_pt);

  FILE *f_gamma = fopen("gamma_new.txt", "wt");
  fprintf(f_gamma,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_gamma,"%6.6f\n", ((float*)gamma_pt)[i]);
  fclose(f_gamma);

  FILE *f_beta = fopen("beta_new.txt", "wt");
  fprintf(f_beta,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_beta,"%6.6f\n", ((float*)beta_pt)[i]);
  fclose(f_beta);

  FILE *f_mean = fopen("mean_new.txt", "wt");
  fprintf(f_mean,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_mean,"%6.6f\n", ((float*)mean_pt)[i]);
  fclose(f_mean);

  FILE *f_var = fopen("var_new.txt", "wt");
  fprintf(f_var,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_var,"%6.6f\n", ((float*)var_pt)[i]);
  fclose(f_var);

  FILE *f_invstd = fopen("invstd_new.txt", "wt");
  fprintf(f_invstd,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_invstd,"%6.6f\n", ((float*)invstd_pt)[i]);
  fclose(f_invstd);

  FILE *f_input = fopen("input_new.txt", "wt");
  fprintf(f_input,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_input,"%6.6f\n", ((float*)input_pt)[i]);
  fclose(f_input);

  if(input_add.numel() > 0) {
    FILE *f_input_add = fopen("input_add_new.txt", "wt");
    fprintf(f_input_add,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
    for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
      fprintf(f_input_add,"%6.6f\n", ((float*)input_add_pt)[i]);
    fclose(f_input_add);
  }

  FILE *f_output = fopen("output_new.txt", "wt");
  fprintf(f_output,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_output,"%6.6f\n", ((float*)output_pt)[i]);
  fclose(f_output);

  FILE *f_relu_mask = fopen("relu_mask_new.txt", "wt");
  fprintf(f_relu_mask,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_relu_mask,"%u\n", ((unsigned char*)relu_mask_pt)[i]);
  fclose(f_relu_mask);

  unsigned char *relumask_uncompressed, *relumask, *eqn_relumask;

  float *out;
  float *naive_inp, *naive_inp_add, *naive_out, *naive_rcpstdev;
  unsigned char *naive_relumask;

  int N  = cfg.fwd_cfg.N;
  int C  = cfg.fwd_cfg.C;
  int H  = cfg.fwd_cfg.H;
  int W  = cfg.fwd_cfg.W;
  int bc = cfg.fwd_cfg.bc;

  naive_fusedbatchnorm_t naive_param;

  naive_param.N = N;
  naive_param.C = C;
  naive_param.H = H;
  naive_param.W = W;
  naive_param.stride_h  = 1;
  naive_param.stride_w  = 1;
  naive_param.norm_type = 0; /* 0: full batchnorm, 1: batch scaling only */
  naive_param.fuse_type = cfg.fwd_cfg.fuse_type; /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */

  relumask              = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*C*H*W, 2097152);
  relumask_uncompressed = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*C*H*W, 2097152);

  out            = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W,   2097152);

  naive_inp      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_out      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_inp_add  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_rcpstdev = (float*) libxsmm_aligned_malloc( sizeof(float)*C,       2097152);
  naive_relumask = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*C*H*W, 2097152);

  tensor_copy_NCHWc_to_NCHW ((float*)input_pt,     naive_inp,     N, C, H, W, bc);
  if(input_add.numel() > 0)
    tensor_copy_NCHWc_to_NCHW ((float*)input_add_pt, naive_inp_add, N, C, H, W, bc);
  else
    zero_buf(naive_inp_add, N*C*H*W);

  naive_fusedbatchnorm_fp(&naive_param, naive_inp, naive_out, naive_inp_add,
                                      (float*)beta_pt, (float*)gamma_pt, cfg.eps, (float*)mean_pt, naive_rcpstdev, (float*)var_pt, naive_relumask);

  tensor_copy_NCHW_to_NCHWc       (naive_out     , out,                   N, C, H, W, bc);
  tensor_copy_NCHW_to_NCHWc_uint8 (naive_relumask, relumask_uncompressed, N, C, H, W, bc);
  /* since naive implementation returnes the mask with 1 char per entry, after changing layout, a compression into bitmask is needed */
  mask_compress_uint8 (relumask_uncompressed, relumask, N*C*H*W);

  FILE *f_output_naive = fopen("output_naive_new.txt", "wt");
  fprintf(f_output_naive,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_output_naive,"%6.6f\n", ((float*)out)[i]);
  fclose(f_output_naive);

  FILE *f_relu_mask_naive = fopen("relu_mask_naive_new.txt", "wt");
  fprintf(f_relu_mask_naive,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_relu_mask_naive,"%u\n", ((unsigned char*)relumask)[i]);
  fclose(f_relu_mask_naive);

  FILE *f_relu_mask_uncompr_naive = fopen("relu_mask_uncompr_naive_new.txt", "wt");
  fprintf(f_relu_mask_uncompr_naive,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_relu_mask_uncompr_naive,"%u\n", ((unsigned char*)relumask_uncompressed)[i]);
  fclose(f_relu_mask_uncompr_naive);

  /* exit(-1); */

#endif /* DUMP_FORWARD_BN */

  return {output, relu_mask};
#endif /* if-else USE_OLD_HANDLE_BN */

}

std::vector<at::Tensor> bnorm_backward_new( pcl_cgbp_bn_config& cfg, torch::Tensor grad_output, torch::Tensor input, torch::Tensor input_add,
                                        torch::Tensor gamma, torch::Tensor output, torch::Tensor mean, torch::Tensor var,
                                        torch::Tensor invstd, torch::Tensor relu_mask)
{
  /* printf("debug: calling bnorm_backward_new with tensor N C H W bc: %d %d %d %d %d\n", cfg.bwd_cfg.N, cfg.bwd_cfg.C, cfg.bwd_cfg.H, cfg.bwd_cfg.W, cfg.bwd_cfg.bc); */

#ifdef USE_OLD_HANDLE_BN
  /* printf("debug: In bnorm_backward_new but with an old implementation \n"); */

  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;

  auto grad_input = input.new_empty(input.sizes());

  at::Tensor grad_input_add;
  if(input_add.numel() > 0)
    grad_input_add = input_add.new_empty(input_add.sizes());

  auto grad_gamma = gamma.new_empty(gamma.sizes());
  auto grad_beta = gamma.new_empty(gamma.sizes());

  libxsmm_dnn_fusedbatchnorm* libxsmm_handle = (libxsmm_dnn_fusedbatchnorm*)(cfg.libxsmm_handle_);
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, gamma, "Gamma");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL, mean, "Mean");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_VARIANCE, var, "Variance");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_RCPSTDDEV, invstd, "Inverse Std");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_RELU_MASK, relu_mask, "ReLU Mask");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, grad_gamma, "GradGamma");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, grad_beta, "GradBeta");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, grad_output, "GradOutput");
  libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, grad_input, "GradInput");

  if(grad_input_add.numel() > 0)
    libxsmm_dnn_fusedbatchnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_ADD, grad_input_add, "GradInputAdd");

  {
    RECORD_FUNCTION("xsmm_bn_bwd", std::vector<c10::IValue>(/*grad_output, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid) );
    }
  }

#ifdef DUMP_BACKWARD_BN
  void *gamma_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(gamma, &gamma_pt);

  void *mean_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(mean, &mean_pt);

  void *var_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(var, &var_pt);

  void *input_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(input, &input_pt);

  void *grad_input_add_pt;
  if(grad_input_add.numel() > 0)
    libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_input_add, &grad_input_add_pt);

  void *output_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(output, &output_pt);

  void *grad_output_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_output, &grad_output_pt);

  void *grad_input_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_input, &grad_input_pt);

  void *grad_gamma_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_gamma, &grad_gamma_pt);

  void *grad_beta_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_beta, &grad_beta_pt);

  void *relu_mask_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(relu_mask, &relu_mask_pt);

  FILE *f_relu_mask = fopen("relu_mask_bwd_old.txt", "wt");
  fprintf(f_relu_mask,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_relu_mask,"%u\n", ((unsigned char*)relu_mask_pt)[i]);
  fclose(f_relu_mask);

  FILE *f_input = fopen("input_bwd_old.txt", "wt");
  fprintf(f_input,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_input,"%6.6f\n", ((float*)input_pt)[i]);
  fclose(f_input);

  FILE *f_gamma = fopen("gamma_bwd_old.txt", "wt");
  fprintf(f_gamma,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_gamma,"%6.6f\n", ((float*)gamma_pt)[i]);
  fclose(f_gamma);

  FILE *f_mean = fopen("mean_bwd_old.txt", "wt");
  fprintf(f_mean,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_mean,"%6.6f\n", ((float*)mean_pt)[i]);
  fclose(f_mean);

  FILE *f_var = fopen("var_bwd_old.txt", "wt");
  fprintf(f_var,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_var,"%6.6f\n", ((float*)var_pt)[i]);
  fclose(f_var);

  FILE *f_output = fopen("output_old.txt", "wt");
  fprintf(f_output,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_output,"%6.6f\n", ((float*)output_pt)[i]);
  fclose(f_output);

  FILE *f_grad_output = fopen("grad_output_old.txt", "wt");
  fprintf(f_grad_output,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_grad_output,"%6.6f\n", ((float*)grad_output_pt)[i]);
  fclose(f_grad_output);

  FILE *f_grad_input = fopen("grad_input_old.txt", "wt");
  fprintf(f_grad_input,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_grad_input,"%6.6f\n", ((float*)grad_input_pt)[i]);
  fclose(f_grad_input);

  if(grad_input_add.numel() > 0) {
    FILE *f_grad_input_add = fopen("grad_input_add_old.txt", "wt");
    fprintf(f_grad_input_add,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
    for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
      fprintf(f_grad_input_add,"%6.6f\n", ((float*)grad_input_add_pt)[i]);
    fclose(f_grad_input_add);
  }

  FILE *f_grad_gamma = fopen("grad_gamma_old.txt", "wt");
  fprintf(f_grad_gamma,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_grad_gamma,"%6.6f\n", ((float*)grad_gamma_pt)[i]);
  fclose(f_grad_gamma);

  FILE *f_grad_beta = fopen("grad_beta_old.txt", "wt");
  fprintf(f_grad_beta,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_grad_beta,"%6.6f\n", ((float*)grad_beta_pt)[i]);
  fclose(f_grad_beta);

  /* exit(-1); */
#endif /* DUMP_BACKWARD_BN */

  return {grad_input, grad_input_add, grad_gamma, grad_beta};

#else /* USE_OLD_HANDLE_BN */

  /* printf("debug: in bnorm_backward_new with new impl\n"); */
/*
  if ( cfg.fwd_cfg.scratch_size > 0 || cfg.bwd_cfg.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( cfg.fwd_cfg.scratch_size, cfg.bwd_cfg.scratch_size);
    init_buf( (float*)(cfg.scratch), (alloc_size)/4, 0, 0 );
  }
*/
  auto grad_input = input.new_empty(input.sizes());

  at::Tensor grad_input_add;
  if(input_add.numel() > 0) {
    //printf("debugging: input_add.numel() is > 0 \n");
    grad_input_add = input_add.new_empty(input_add.sizes());
  } else {
    //printf("debugging: input_add.numel() is 0 \n");
    grad_input_add = at::zeros({0}, input.options());
  }

  auto grad_gamma = gamma.new_empty(gamma.sizes());
  auto grad_beta = gamma.new_empty(gamma.sizes());

  void *grad_output_pt = NULL, *input_pt = NULL, *mean_pt = NULL, *var_pt = NULL, *gamma_pt = NULL,
         *grad_input_pt = NULL, *grad_input_add_pt = NULL, *grad_gamma_pt = NULL, *grad_beta_pt = NULL, *relu_mask_pt = NULL;

  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_output,    &grad_output_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(input,          &input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(mean,           &mean_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(var,            &var_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(gamma,          &gamma_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_input,     &grad_input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_gamma,     &grad_gamma_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_beta,      &grad_beta_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(relu_mask,      &relu_mask_pt);

  if(grad_input_add.numel() > 0)
    libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_input_add, &grad_input_add_pt);

#ifdef RECORD_FUNCTIONS_MACRO
  std::string label{"xsmm_bn_bwd_new"};
  label += "_";
  label += std::to_string(cfg.bwd_cfg.N);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.C);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.H);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.W);
#endif

  {
#ifdef RECORD_FUNCTIONS_MACRO
    //RECORD_FUNCTION("xsmm_bn_bwd_new", std::vector<c10::IValue>(/*grad_output, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
    RECORD_FUNCTION(label, std::vector<c10::IValue>(/*grad_output, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      if (cfg.dtype == 0)
        libxsmm_dnn_bn_bwd_exec_f32( cfg.bwd_cfg, (float*)grad_output_pt, (float*)input_pt, (float*)mean_pt, (float*)var_pt, (float*)gamma_pt, (const unsigned char*)relu_mask_pt,
                            (float*)grad_input_pt, (float*)grad_input_add_pt, (float*)grad_gamma_pt, (float*)grad_beta_pt, cfg.eps,
                            0, tid, cfg.scratch, LIBXSMM_DNN_BN_FULL_NORM); /* last argument can be changed for eval mode to LIBXSMM_DNN_BN_SCALE_ONLY for perf opt of evaluation */
      else
        libxsmm_dnn_bn_bwd_exec_bf16( cfg.bwd_cfg, (libxsmm_bfloat16*)grad_output_pt, (libxsmm_bfloat16*)input_pt, (float*)mean_pt, (float*)var_pt, (float*)gamma_pt, (const unsigned char*)relu_mask_pt,
                            (libxsmm_bfloat16*)grad_input_pt, (libxsmm_bfloat16*)grad_input_add_pt, (float*)grad_gamma_pt, (float*)grad_beta_pt, cfg.eps,
                            0, tid, cfg.scratch, LIBXSMM_DNN_BN_FULL_NORM); /* last argument can be changed for eval mode to LIBXSMM_DNN_BN_SCALE_ONLY for perf opt of evaluation */
    }
  }

#ifdef DUMP_BACKWARD_BN
  printf("fuse type = %d\n", (int)cfg.bwd_cfg.fuse_type);

  FILE *f_relu_mask = fopen("relu_mask_bwd_new.txt", "wt");
  fprintf(f_relu_mask,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_relu_mask,"%u\n", ((unsigned char*)relu_mask_pt)[i]);
  fclose(f_relu_mask);

  FILE *f_input = fopen("input_bwd_new.txt", "wt");
  fprintf(f_input,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_input,"%6.6f\n", ((float*)input_pt)[i]);
  fclose(f_input);

  FILE *f_gamma = fopen("gamma_bwd_new.txt", "wt");
  fprintf(f_gamma,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_gamma,"%6.6f\n", ((float*)gamma_pt)[i]);
  fclose(f_gamma);

  FILE *f_mean = fopen("mean_bwd_new.txt", "wt");
  fprintf(f_mean,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_mean,"%6.6f\n", ((float*)mean_pt)[i]);
  fclose(f_mean);

  FILE *f_var = fopen("var_bwd_new.txt", "wt");
  fprintf(f_var,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_var,"%6.6f\n", ((float*)var_pt)[i]);
  fclose(f_var);

  void *output_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(output,    &output_pt);

  FILE *f_output = fopen("output_new.txt", "wt");
  fprintf(f_output,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_output,"%6.6f\n", ((float*)output_pt)[i]);
  fclose(f_output);

  FILE *f_grad_output = fopen("grad_output_new.txt", "wt");
  fprintf(f_grad_output,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_grad_output,"%6.6f\n", ((float*)grad_output_pt)[i]);
  fclose(f_grad_output);

  FILE *f_grad_input = fopen("grad_input_new.txt", "wt");
  fprintf(f_grad_input,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_grad_input,"%6.6f\n", ((float*)grad_input_pt)[i]);
  fclose(f_grad_input);

  if(grad_input_add.numel() > 0) {
    FILE *f_grad_input_add = fopen("grad_input_add_new.txt", "wt");
    fprintf(f_grad_input_add,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
    for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
      fprintf(f_grad_input_add,"%6.6f\n", ((float*)grad_input_add_pt)[i]);
    fclose(f_grad_input_add);
  }

  FILE *f_grad_gamma = fopen("grad_gamma_new.txt", "wt");
  fprintf(f_grad_gamma,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_grad_gamma,"%6.6f\n", ((float*)grad_gamma_pt)[i]);
  fclose(f_grad_gamma);

  FILE *f_grad_beta = fopen("grad_beta_new.txt", "wt");
  fprintf(f_grad_beta,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_grad_beta,"%6.6f\n", ((float*)grad_beta_pt)[i]);
  fclose(f_grad_beta);

  /* exit(-1); */

#endif /* DUMP_BACKWARD_BN */

  return {grad_input, grad_input_add, grad_gamma, grad_beta};
#endif /* if-else USE_OLD_HANDLE_BN */
}

#endif /* NEW_BATCHNORM */


#ifdef NEW_GROUPNORM

typedef struct pcl_cgbp_gn_config {
  libxsmm_dnn_gn_fwd_config fwd_cfg;
  libxsmm_dnn_gn_bwd_config bwd_cfg;
  int              dtype;               /* 0 for fp32, 1 for bf16 */
  void*            scratch;
  float            eps;
#ifdef USE_OLD_HANDLE_GN
  void*            libxsmm_handle_;
#endif
} pcl_cgbp_gn_config;


int gnorm_get_c_block( int C /*, datatype as an int flag? */ )
{
  libxsmm_blasint bc = CHANNEL_BLOCK_SIZE; /* hardcoded for now */

  if (C % bc != 0)
    bc = C;

  return bc;
}


pcl_cgbp_gn_config gnorm_setup_new(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint G,
                              libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in, libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                              float eps, int fuse_type_int, int dtype_int ) {
  pcl_cgbp_gn_config res;

  libxsmm_dnn_gn_fuse fuse_type_enum = LIBXSMM_DNN_GN_FUSE_NONE;

  switch(fuse_type_int) {
    case 0:
      fuse_type_enum = LIBXSMM_DNN_GN_FUSE_NONE;
      break;
    case 1:
      fuse_type_enum = LIBXSMM_DNN_GN_FUSE_RELU;
      break;
    case 2:
      fuse_type_enum = LIBXSMM_DNN_GN_FUSE_ELTWISE;
      break;
    case 3:
      fuse_type_enum = LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU;
      break;
    case 4:
      fuse_type_enum = LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK;
      break;
    case 5:
      fuse_type_enum = LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK;
      break;
    default:
      printf("Unrecognized fuse_type (int) = %d in gnorm_setup_new (assuming FUSE_NONE)\n", fuse_type_int);
      break;
  }

  libxsmm_blasint bc      = gnorm_get_c_block(C);
  libxsmm_blasint threads = (libxsmm_blasint)omp_get_max_threads();

  libxsmm_datatype gn_dtype_in  = (dtype_int == 0 ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16);
  libxsmm_datatype gn_dtype_out = gn_dtype_in;
  libxsmm_datatype gn_dtype_comp = LIBXSMM_DATATYPE_F32;
  res.dtype = dtype_int;

  res.fwd_cfg = setup_libxsmm_dnn_gn_fwd(N, C, H, W, G, bc, pad_h_in, pad_w_in, pad_h_out, pad_w_out, threads, fuse_type_enum, gn_dtype_in, gn_dtype_out, gn_dtype_comp);
  res.bwd_cfg = setup_libxsmm_dnn_gn_bwd(N, C, H, W, G, bc, pad_h_in, pad_w_in, pad_h_out, pad_w_out, threads, fuse_type_enum, gn_dtype_in, gn_dtype_out, gn_dtype_comp);

  /* allocate and bind scratch */
  void *scratch = NULL;
  if ( res.fwd_cfg.scratch_size > 0 || res.bwd_cfg.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( res.fwd_cfg.scratch_size, res.bwd_cfg.scratch_size);
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    //init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
    zero_buf((float*)scratch, (alloc_size)/4);
  }

  res.scratch = scratch;
  res.eps     = eps;

#ifdef USE_OLD_HANDLE_GN
  bool relu = false, eltwise = false/*, train = false*/;
  if (fuse_type_enum == LIBXSMM_DNN_GN_FUSE_RELU || fuse_type_enum == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || fuse_type_enum == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
    relu = true;
  if (fuse_type_enum == LIBXSMM_DNN_GN_FUSE_ELTWISE || fuse_type_enum == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || fuse_type_enum == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
    eltwise = true;
  /*if (fuse_type_enum == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || fuse_type_enum == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
    train = true;
  */
  res.libxsmm_handle_ = fusedgroupnorm_create_handle(N, C, H, W, G, 1 /*int dtype*/, relu, eltwise);
#endif

  return res;
}

void gnorm_setup_destroy_new(pcl_cgbp_gn_config cfg)
{
  if (cfg.scratch)
      libxsmm_free(cfg.scratch);
  cfg.scratch = NULL;

  destroy_libxsmm_dnn_gn_fwd(&cfg.fwd_cfg);
  destroy_libxsmm_dnn_gn_bwd(&cfg.bwd_cfg);

#ifdef USE_OLD_HANDLE_GN
  fusedgroupnorm_destroy_handle(cfg.libxsmm_handle_);
  cfg.libxsmm_handle_ = NULL;
#endif

  return;
}

/* invstd is unused */
std::vector<at::Tensor> gnorm_forward_new( pcl_cgbp_gn_config& cfg, torch::Tensor input, torch::Tensor input_add, torch::Tensor gamma, torch::Tensor beta,
                                           torch::Tensor mean, torch::Tensor var, torch::Tensor invstd, std::vector<long> output_size)
{
  /* printf("calling gnorm_forward_new with tensor N C H W G bc: %d %d %d %d %d %d %d\n", cfg.fwd_cfg.N, cfg.fwd_cfg.C, cfg.fwd_cfg.H, cfg.fwd_cfg.W, cfg.fwd_cfg.G, cfg.fwd_cfg.bc, (int)cfg.fwd_cfg.fuse_type); */

  /* printf("In gnorm_forward_new with new impl\n"); */

//  at::Tensor mean = at::empty(stats_size, torch::TensorOptions().dtype(dt_map[1]));
//  at::Tensor var = at::empty(stats_size, torch::TensorOptions().dtype(dt_map[1]));
//  at::Tensor invstd = at::empty(stats_size, torch::TensorOptions().dtype(dt_map[1]));

  at::Tensor relu_mask = at::empty(output_size, torch::TensorOptions().dtype(dt_map[7]));
  at::Tensor output;
  output = input.new_empty(output_size);

  void *input_pt = NULL, *input_add_pt = NULL, *gamma_pt = NULL, *beta_pt = NULL,
        *mean_pt = NULL, *var_pt = NULL, *output_pt = NULL, *relu_mask_pt = NULL;

  libxsmm_tpp_convert_at_tensor_to_raw_helper(input,     &input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(gamma,     &gamma_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(beta,      &beta_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(mean,      &mean_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(var,       &var_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(output,    &output_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(relu_mask, &relu_mask_pt);

  /* FIXME: Is this needed at all? */
  /* init_buf( (float*)(invstd_pt), cfg.fwd_cfg.N * cfg.fwd_cfg.G, 0, 0 ); */

  if(input_add.numel() > 0)
    libxsmm_tpp_convert_at_tensor_to_raw_helper(input_add, &input_add_pt);

#ifdef RECORD_FUNCTIONS_MACRO
  std::string label{"xsmm_gn_fwd_new"};
  label += "_";
  label += std::to_string(cfg.fwd_cfg.N);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.C);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.H);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.W);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.G);
#endif
  {
#ifdef RECORD_FUNCTIONS_MACRO
    //RECORD_FUNCTION("xsmm_gn_fwd_new", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
    RECORD_FUNCTION(label, std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      if (cfg.dtype == 0)
        libxsmm_dnn_gn_fwd_exec_f32 ( cfg.fwd_cfg, (const float*)input_pt, (const float*)input_add_pt, (const float*)gamma_pt, (const float*)beta_pt, (float*)mean_pt, (float*)var_pt,
                            (float*)output_pt, (unsigned char*)relu_mask_pt, cfg.eps, 0, tid, cfg.scratch );
      else
        libxsmm_dnn_gn_fwd_exec_bf16( cfg.fwd_cfg, (const libxsmm_bfloat16*)input_pt, (const libxsmm_bfloat16*)input_add_pt, (const float*)gamma_pt, (const float*)beta_pt, (float*)mean_pt, (float*)var_pt,
                            (libxsmm_bfloat16*)output_pt, (unsigned char*)relu_mask_pt, cfg.eps, 0, tid, cfg.scratch );
    }
  }

#ifdef DUMP_FORWARD_GN

  FILE *f_gamma = fopen("gamma_new.txt", "wt");
  fprintf(f_gamma,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_gamma,"%6.6f\n", ((float*)gamma_pt)[i]);
  fclose(f_gamma);

  FILE *f_beta = fopen("beta_new.txt", "wt");
  fprintf(f_beta,"%d\n", cfg.fwd_cfg.C);
  for (int i = 0; i < cfg.fwd_cfg.C; i++)
    fprintf(f_beta,"%6.6f\n", ((float*)beta_pt)[i]);
  fclose(f_beta);

  FILE *f_mean = fopen("mean_new.txt", "wt");
  fprintf(f_mean,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.G);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.G; i++)
    fprintf(f_mean,"%6.6f\n", ((float*)mean_pt)[i]);
  fclose(f_mean);

  FILE *f_var = fopen("var_new.txt", "wt");
  fprintf(f_var,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.G);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.G; i++)
    fprintf(f_var,"%6.6f\n", ((float*)var_pt)[i]);
  fclose(f_var);

  /*
  FILE *f_invstd = fopen("invstd_new.txt", "wt");
  fprintf(f_invstd,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.G);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.G; i++)
    fprintf(f_invstd,"%6.6f\n", ((float*)invstd_pt)[i]);
  fclose(f_invstd);
  */

  FILE *f_input = fopen("input_new.txt", "wt");
  fprintf(f_input,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_input,"%6.6f\n", ((float*)input_pt)[i]);
  fclose(f_input);

  if(input_add.numel() > 0) {
    FILE *f_input_add = fopen("input_add_new.txt", "wt");
    fprintf(f_input_add,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
    for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
      fprintf(f_input_add,"%6.6f\n", ((float*)input_add_pt)[i]);
    fclose(f_input_add);
  }

  FILE *f_output = fopen("output_new.txt", "wt");
  fprintf(f_output,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_output,"%6.6f\n", ((float*)output_pt)[i]);
  fclose(f_output);

  FILE *f_relu_mask = fopen("relu_mask_new.txt", "wt");
  fprintf(f_relu_mask,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_relu_mask,"%u\n", ((unsigned char*)relu_mask_pt)[i]);
  fclose(f_relu_mask);

  unsigned char *relumask_uncompressed, *relumask, *eqn_relumask;

  float *out;
  float *naive_inp, *naive_inp_add, *naive_out, *naive_rcpstdev;
  unsigned char *naive_relumask;

  int N  = cfg.fwd_cfg.N;
  int C  = cfg.fwd_cfg.C;
  int H  = cfg.fwd_cfg.H;
  int W  = cfg.fwd_cfg.W;
  int G  = cfg.fwd_cfg.G;
  int bc = cfg.fwd_cfg.bc;

  naive_fusedgroupnorm_t naive_param;

  naive_param.N = N;
  naive_param.C = C;
  naive_param.H = H;
  naive_param.W = W;
  naive_param.G = G;
  naive_param.stride_h  = 1;
  naive_param.stride_w  = 1;
  naive_param.fuse_type = cfg.fwd_cfg.fuse_type; /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */

  relumask              = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*C*H*W, 2097152);
  relumask_uncompressed = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*C*H*W, 2097152);

  out            = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W,   2097152);

  naive_inp      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_out      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_inp_add  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_rcpstdev = (float*) libxsmm_aligned_malloc( sizeof(float)*C,       2097152);
  naive_relumask = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*C*H*W, 2097152);

  tensor_copy_NCHWc_to_NCHW ((float*)input_pt,     naive_inp,     N, C, H, W, bc);
  if(input_add.numel() > 0)
    tensor_copy_NCHWc_to_NCHW ((float*)input_add_pt, naive_inp_add, N, C, H, W, bc);
  else
    zero_buf(naive_inp_add, N*C*H*W);

  naive_fusedgroupnorm_fp(&naive_param, naive_inp, naive_out, naive_inp_add,
                                      (float*)beta_pt, (float*)gamma_pt, cfg.eps, (float*)mean_pt, naive_rcpstdev, (float*)var_pt, naive_relumask);

  tensor_copy_NCHW_to_NCHWc       (naive_out     , out,                   N, C, H, W, bc);
  tensor_copy_NCHW_to_NCHWc_uint8 (naive_relumask, relumask_uncompressed, N, C, H, W, bc);
  /* since naive implementation returnes the mask with 1 char per entry, after changing layout, a compression into bitmask is needed */
  mask_compress_uint8 (relumask_uncompressed, relumask, N*C*H*W);

  FILE *f_output_naive = fopen("output_naive_new.txt", "wt");
  fprintf(f_output_naive,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_output_naive,"%6.6f\n", ((float*)out)[i]);
  fclose(f_output_naive);

  FILE *f_relu_mask_naive = fopen("relu_mask_naive_new.txt", "wt");
  fprintf(f_relu_mask_naive,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_relu_mask_naive,"%u\n", ((unsigned char*)relumask)[i]);
  fclose(f_relu_mask_naive);

  FILE *f_relu_mask_uncompr_naive = fopen("relu_mask_uncompr_naive_new.txt", "wt");
  fprintf(f_relu_mask_uncompr_naive,"%d\n", cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W);
  for (int i = 0; i < cfg.fwd_cfg.N * cfg.fwd_cfg.C * cfg.fwd_cfg.H * cfg.fwd_cfg.W; i++)
    fprintf(f_relu_mask_uncompr_naive,"%u\n", ((unsigned char*)relumask_uncompressed)[i]);
  fclose(f_relu_mask_uncompr_naive);

  /* exit(-1); */

#endif /* DUMP_FORWARD_GN */

  return {output, relu_mask};

}

std::vector<at::Tensor> gnorm_backward_new( pcl_cgbp_gn_config& cfg, torch::Tensor grad_output, torch::Tensor input, torch::Tensor input_add,
                                        torch::Tensor gamma, torch::Tensor output, torch::Tensor mean, torch::Tensor var,
                                        torch::Tensor invstd, torch::Tensor relu_mask)
{
  /* printf("calling gnorm_backward_new with tensor N C H W G bc: %d %d %d %d %d %d\n", cfg.bwd_cfg.N, cfg.bwd_cfg.C, cfg.bwd_cfg.H, cfg.bwd_cfg.W, cfg.bwd_cfg.G, cfg.bwd_cfg.bc); */

#ifdef USE_OLD_HANDLE_GN
//#if 0
  printf("In gnorm_backward_new but with an old implementation \n");

  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;

  auto grad_input = input.new_empty(input.sizes());

  at::Tensor grad_input_add;
  if(input_add.numel() > 0)
    grad_input_add = input_add.new_empty(input_add.sizes());

  auto grad_gamma = gamma.new_empty(gamma.sizes());
  auto grad_beta  = gamma.new_empty(gamma.sizes());

  libxsmm_dnn_fusedgroupnorm* libxsmm_handle = (libxsmm_dnn_fusedgroupnorm*)(cfg.libxsmm_handle_);
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, gamma, "Gamma");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL, mean, "Mean");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_VARIANCE, var, "Variance");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_CHANNEL_RCPSTDDEV, invstd, "Inverse Std");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_RELU_MASK, relu_mask, "ReLU Mask");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, grad_gamma, "GradGamma");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, grad_beta, "GradBeta");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, grad_output, "GradOutput");
  libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, grad_input, "GradInput");

  if(grad_input_add.numel() > 0)
    libxsmm_dnn_fusedgroupnorm_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_ADD, grad_input_add, "GradInputAdd");
  {
    RECORD_FUNCTION("xsmm_gn_bwd", std::vector<c10::IValue>(/*grad_output, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedgroupnorm_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid) );
    }
  }

#ifdef DUMP_BACKWARD_GN
  //printf("fuse type = %d\n", (int)libxsmm_handle.fuse_type);

  void *gamma_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(gamma, &gamma_pt);

  void *mean_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(mean, &mean_pt);

  void *var_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(var, &var_pt);

  void *input_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(input, &input_pt);

  void *grad_input_add_pt;
  if(grad_input_add.numel() > 0)
    libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_input_add, &grad_input_add_pt);

  void *output_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(output, &output_pt);

  void *invstd_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(invstd, &invstd_pt);

  void *grad_output_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_output, &grad_output_pt);

  void *grad_input_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_input, &grad_input_pt);

  void *grad_gamma_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_gamma, &grad_gamma_pt);

  void *grad_beta_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_beta, &grad_beta_pt);

  void *relu_mask_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(relu_mask, &relu_mask_pt);

  FILE *f_relu_mask = fopen("relu_mask_bwd_old.txt", "wt");
  fprintf(f_relu_mask,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W; i++)
    fprintf(f_relu_mask,"%u\n", ((unsigned char*)relu_mask_pt)[i]);
  fclose(f_relu_mask);

  FILE *f_input = fopen("input_bwd_old.txt", "wt");
  fprintf(f_input,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W; i++)
    fprintf(f_input,"%6.6f\n", ((float*)input_pt)[i]);
  fclose(f_input);

  FILE *f_gamma = fopen("gamma_bwd_old.txt", "wt");
  fprintf(f_gamma,"%d\n", cfg.bwd_cfg.C);
  for (int i = 0; i < cfg.bwd_cfg.C; i++)
    fprintf(f_gamma,"%6.6f\n", ((float*)gamma_pt)[i]);
  fclose(f_gamma);

  FILE *f_mean = fopen("mean_bwd_old.txt", "wt");
  fprintf(f_mean,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.G);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.G; i++)
    fprintf(f_mean,"%6.6f\n", ((float*)mean_pt)[i]);
  fclose(f_mean);

  FILE *f_var = fopen("var_bwd_old.txt", "wt");
  fprintf(f_var,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.G);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.G; i++)
    fprintf(f_var,"%6.6f\n", ((float*)var_pt)[i]);
  fclose(f_var);

  FILE *f_invstd = fopen("invstd_bwd_old.txt", "wt");
  fprintf(f_invstd,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.G);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.G; i++)
    fprintf(f_invstd,"%6.6f\n", ((float*)invstd_pt)[i]);
  fclose(f_invstd);

  FILE *f_output = fopen("output_old.txt", "wt");
  fprintf(f_output,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W; i++)
    fprintf(f_output,"%6.6f\n", ((float*)output_pt)[i]);
  fclose(f_output);

  FILE *f_grad_output = fopen("grad_output_old.txt", "wt");
  fprintf(f_grad_output,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W; i++)
    fprintf(f_grad_output,"%6.6f\n", ((float*)grad_output_pt)[i]);
  fclose(f_grad_output);

  FILE *f_grad_input = fopen("grad_input_old.txt", "wt");
  fprintf(f_grad_input,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W; i++)
    fprintf(f_grad_input,"%6.6f\n", ((float*)grad_input_pt)[i]);
  fclose(f_grad_input);

  if(grad_input_add.numel() > 0) {
    FILE *f_grad_input_add = fopen("grad_input_add_old.txt", "wt");
    fprintf(f_grad_input_add,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W);
    for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W; i++)
      fprintf(f_grad_input_add,"%6.6f\n", ((float*)grad_input_add_pt)[i]);
    fclose(f_grad_input_add);
  }

  FILE *f_grad_gamma = fopen("grad_gamma_old.txt", "wt");
  fprintf(f_grad_gamma,"%d\n", cfg.bwd_cfg.C);
  for (int i = 0; i < cfg.bwd_cfg.C; i++)
    fprintf(f_grad_gamma,"%6.6f\n", ((float*)grad_gamma_pt)[i]);
  fclose(f_grad_gamma);

  FILE *f_grad_beta = fopen("grad_beta_old.txt", "wt");
  fprintf(f_grad_beta,"%d\n", cfg.bwd_cfg.C);
  for (int i = 0; i < cfg.bwd_cfg.C; i++)
    fprintf(f_grad_beta,"%6.6f\n", ((float*)grad_beta_pt)[i]);
  fclose(f_grad_beta);

  /* exit(-1); */
#endif /* DUMP_BACKWARD_GN */

  return {grad_input, grad_input_add, grad_gamma, grad_beta};

#else /* USE_OLD_HANDLE_GN */

  /* printf("In gnorm_backward_new with new impl\n"); */

  auto grad_input = input.new_empty(input.sizes());

  at::Tensor grad_input_add;
  if(input_add.numel() > 0)
    grad_input_add = input_add.new_empty(input_add.sizes());

  auto grad_gamma = gamma.new_empty(gamma.sizes());
  auto grad_beta = gamma.new_empty(gamma.sizes());

  void *grad_output_pt = NULL, *input_pt = NULL, *mean_pt = NULL, *var_pt = NULL, *gamma_pt = NULL,
        *grad_input_pt = NULL, *grad_input_add_pt = NULL, *grad_gamma_pt = NULL, *grad_beta_pt = NULL, *relu_mask_pt = NULL;

  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_output,    &grad_output_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(input,          &input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(mean,           &mean_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(var,            &var_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(gamma,          &gamma_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_input,     &grad_input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_gamma,     &grad_gamma_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_beta,      &grad_beta_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(relu_mask,      &relu_mask_pt);

  if(grad_input_add.numel() > 0)
      libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_input_add, &grad_input_add_pt);

#ifdef RECORD_FUNCTIONS_MACRO
  std::string label{"xsmm_gn_bwd_new"};
  label += "_";
  label += std::to_string(cfg.bwd_cfg.N);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.C);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.H);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.W);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.G);
#endif

  {
#ifdef RECORD_FUNCTIONS_MACRO
    //RECORD_FUNCTION("xsmm_gn_bwd_new", std::vector<c10::IValue>(/*grad_output, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
    RECORD_FUNCTION(label, std::vector<c10::IValue>(/*grad_output, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      if (cfg.dtype == 0)
        libxsmm_dnn_gn_bwd_exec_f32 ( cfg.bwd_cfg, (float*)grad_output_pt, (float*)input_pt, (float*)mean_pt, (float*)var_pt, (float*)gamma_pt, (const unsigned char*)relu_mask_pt,
                            (float*)grad_input_pt, (float*)grad_input_add_pt, (float*)grad_gamma_pt, (float*)grad_beta_pt, cfg.eps,
                             0, tid, cfg.scratch);
      else
        libxsmm_dnn_gn_bwd_exec_bf16( cfg.bwd_cfg, (libxsmm_bfloat16*)grad_output_pt, (libxsmm_bfloat16*)input_pt, (float*)mean_pt, (float*)var_pt, (float*)gamma_pt, (const unsigned char*)relu_mask_pt,
                            (libxsmm_bfloat16*)grad_input_pt, (libxsmm_bfloat16*)grad_input_add_pt, (float*)grad_gamma_pt, (float*)grad_beta_pt, cfg.eps,
                             0, tid, cfg.scratch);
    }
  }

#ifdef DUMP_BACKWARD_GN
  printf("fuse type = %d\n", (int)cfg.bwd_cfg.fuse_type);

  FILE *f_relu_mask = fopen("relu_mask_bwd_new.txt", "wt");
  fprintf(f_relu_mask,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W; i++)
    fprintf(f_relu_mask,"%u\n", ((unsigned char*)relu_mask_pt)[i]);
  fclose(f_relu_mask);

  FILE *f_input = fopen("input_bwd_new.txt", "wt");
  fprintf(f_input,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W; i++)
    fprintf(f_input,"%6.6f\n", ((float*)input_pt)[i]);
  fclose(f_input);

  FILE *f_gamma = fopen("gamma_bwd_new.txt", "wt");
  fprintf(f_gamma,"%d\n", cfg.bwd_cfg.C);
  for (int i = 0; i < cfg.bwd_cfg.C; i++)
    fprintf(f_gamma,"%6.6f\n", ((float*)gamma_pt)[i]);
  fclose(f_gamma);

  FILE *f_mean = fopen("mean_bwd_new.txt", "wt");
  fprintf(f_mean,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.G);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.G; i++)
    fprintf(f_mean,"%6.6f\n", ((float*)mean_pt)[i]);
  fclose(f_mean);

  FILE *f_var = fopen("var_bwd_new.txt", "wt");
  fprintf(f_var,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.G);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.G; i++)
    fprintf(f_var,"%6.6f\n", ((float*)var_pt)[i]);
  fclose(f_var);

  void *invstd_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(invstd, &invstd_pt);

  FILE *f_invstd = fopen("invstd_bwd_new.txt", "wt");
  fprintf(f_invstd,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.G);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.G; i++)
    fprintf(f_invstd,"%6.6f\n", ((float*)invstd_pt)[i]);
  fclose(f_invstd);

  void *output_pt;
  libxsmm_tpp_convert_at_tensor_to_raw_helper(output,    &output_pt);

  FILE *f_output = fopen("output_new.txt", "wt");
  fprintf(f_output,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W; i++)
    fprintf(f_output,"%6.6f\n", ((float*)output_pt)[i]);
  fclose(f_output);

  FILE *f_grad_output = fopen("grad_output_new.txt", "wt");
  fprintf(f_grad_output,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W; i++)
    fprintf(f_grad_output,"%6.6f\n", ((float*)grad_output_pt)[i]);
  fclose(f_grad_output);

  FILE *f_grad_input = fopen("grad_input_new.txt", "wt");
  fprintf(f_grad_input,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W);
  for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W; i++)
    fprintf(f_grad_input,"%6.6f\n", ((float*)grad_input_pt)[i]);
  fclose(f_grad_input);

  if(grad_input_add.numel() > 0) {
    FILE *f_grad_input_add = fopen("grad_input_add_new.txt", "wt");
    fprintf(f_grad_input_add,"%d\n", cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W);
    for (int i = 0; i < cfg.bwd_cfg.N * cfg.bwd_cfg.C * cfg.bwd_cfg.H * cfg.bwd_cfg.W; i++)
      fprintf(f_grad_input_add,"%6.6f\n", ((float*)grad_input_add_pt)[i]);
    fclose(f_grad_input_add);
  }

  FILE *f_grad_gamma = fopen("grad_gamma_new.txt", "wt");
  fprintf(f_grad_gamma,"%d\n", cfg.bwd_cfg.C);
  for (int i = 0; i < cfg.bwd_cfg.C; i++)
    fprintf(f_grad_gamma,"%6.6f\n", ((float*)grad_gamma_pt)[i]);
  fclose(f_grad_gamma);

  FILE *f_grad_beta = fopen("grad_beta_new.txt", "wt");
  fprintf(f_grad_beta,"%d\n", cfg.bwd_cfg.C);
  for (int i = 0; i < cfg.bwd_cfg.C; i++)
    fprintf(f_grad_beta,"%6.6f\n", ((float*)grad_beta_pt)[i]);
  fclose(f_grad_beta);

#endif /* DUMP_BACKWARD_GN */

  return {grad_input, grad_input_add, grad_gamma, grad_beta};
#endif /* if-else USE_OLD_HANDLE_GN */
}

#endif /* NEW_GROUPNORM */

#ifdef NEW_CONV

  #define HARDCODED_BC (64)
  #define HARDCODED_BK (64)

typedef struct pcl_cgbp_conv_config {
  libxsmm_dnn_conv_config cnn_cfg;
  int                     dtype;             /* 0 for fp32, 1 for bf16 */
  void*                   scratch;
#ifdef USE_OLD_HANDLE_CONV
  void*                   libxsmm_handle_;
#endif
} pcl_cgbp_conv_config;

#ifndef PART_OF_EXTENSIONS
/* Returns a vector of size 3: {C_block, K_block, lp_block} */
std::vector<int> conv_get_feature_map_blocks( int C, int K, int dtype_int )
{
    std::vector<int> res;
    int C_block, K_block, fm_lp_block;

    libxsmm_datatype dtype_in  = (dtype_int == 0 ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16);
    libxsmm_datatype dtype_out = dtype_in;

    libxsmm_dnn_conv_get_feature_map_blocks( C, K, &C_block, &K_block, &fm_lp_block, dtype_in, dtype_out, HARDCODED_BC, HARDCODED_BK );

    res.push_back(C_block);
    res.push_back(K_block);
    res.push_back(fm_lp_block);

    return res;
}
#else
std::vector<int> conv_get_feature_map_blocks( int C, int K, int dtype_int );
#endif

pcl_cgbp_conv_config conv_setup_new(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint K, libxsmm_blasint R, libxsmm_blasint S,
                              libxsmm_blasint pad_h, libxsmm_blasint pad_w, libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in, libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                              libxsmm_blasint stride, int dtype_int, int bc_or_negative, int bk_or_negative )
{
  pcl_cgbp_conv_config res;

  libxsmm_blasint bc, bk;
  if (bc_or_negative < 0) {
//  if (C % 64 == 0)
    bc = CHANNEL_BLOCK_SIZE; /* hardcoded for now */
//  else
//    bc = C;
  } else {
    bc = bc_or_negative;
  }
  if (bc_or_negative < 0) {
//  if (K % 64 == 0)
    bk = K_BLOCK_SIZE;
//  else
//    bk = K;
  //libxsmm_blasint bk      = K_BLOCK_SIZE;       /* hardcoded for now */
  } else {
    bk = bk_or_negative;
  }
  libxsmm_blasint threads = (libxsmm_blasint)omp_get_max_threads();

  /*printf("debug: calling conv_setup_new with tensor N H W C K R S padding stride bc bk: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                                                                                          N, H, W, C, K, R, S,
                                                                                          pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out, stride, bc, bk);*/

  libxsmm_dnn_conv_eltwise_fuse fuse_type = LIBXSMM_DNN_CONV_ELTWISE_FUSE_NONE; /* FIXME: to be changed later? */

  libxsmm_datatype cnn_dtype_in  = (dtype_int == 0 ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16);
  libxsmm_datatype cnn_dtype_out = cnn_dtype_in;

  res.dtype = dtype_int;

  libxsmm_blasint overwrite_output    = 1; /* hardcoded for now */
  libxsmm_blasint avoid_bwd_wt_trans  = 0; /* hardcoded for now */
  libxsmm_blasint zero_fwd_output_rim = 0; /* hardcoded for now */

  /* memset( &res,  0, sizeof(res)); */

  res.cnn_cfg = setup_libxsmm_dnn_conv(cnn_dtype_in, cnn_dtype_out, N, H, W, C, K, R, S, stride, stride, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out, bc, bk, threads,
                                       fuse_type, overwrite_output, avoid_bwd_wt_trans, zero_fwd_output_rim);

  /* allocate and bind scratch */
  void *scratch = NULL;
  if ( (res.cnn_cfg.scratch_size) > 0 ) {
    size_t alloc_size = res.cnn_cfg.scratch_size;
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    //init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
    zero_buf((float*)scratch, (alloc_size)/4);
  }

  res.scratch = scratch;

#ifdef USE_OLD_HANDLE_CONV
  res.libxsmm_handle_ = conv_create_handle(N, C, H, W, K, R, S, padding, stride, 1);
#endif

  return res;
}

void conv_setup_destroy_new(pcl_cgbp_conv_config cfg)
{
  if (cfg.scratch)
    libxsmm_free(cfg.scratch);
  cfg.scratch = NULL;

  destroy_libxsmm_dnn_conv(&cfg.cnn_cfg);

#ifdef USE_OLD_HANDLE_CONV
  conv_destroy_handle(cfg.libxsmm_handle_);
  cfg.libxsmm_handle_ = NULL;
#endif

  return;
}

at::Tensor conv_forward_new(pcl_cgbp_conv_config &cfg, torch::Tensor input, torch::Tensor weight, std::vector<long> dim_size)
{
  /* printf("debug: calling conv_forward_new with tensor N H W C K R S: %d %d %d %d %d %d %d\n", cfg.cnn_cfg.N, cfg.cnn_cfg.H, cfg.cnn_cfg.W, cfg.cnn_cfg.C, cfg.cnn_cfg.K, cfg.cnn_cfg.R, cfg.cnn_cfg.S); */
#ifdef USE_OLD_HANDLE_CONV
  /* printf("debug: in conv_forward_new but with an old implementation \n"); */

  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_layer* libxsmm_handle = (libxsmm_dnn_layer*)(cfg.libxsmm_handle_);

  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, weight, "Weight");

  at::Tensor output = input.new_empty(dim_size);
  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, output, "Output");

  {
    RECORD_FUNCTION("xsmm_conv_fwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
    }
  }
  return output;
#else
//  printf("In conv_forward_new with a new implementation \n");
//  printf("cfg blocksifm = %d blocksofm = %d ifmblock = %d ofmblock = %d \n", cfg.cnn_cfg.blocksifm, cfg.cnn_cfg.blocksofm, cfg.cnn_cfg.ifmblock, cfg.cnn_cfg.ofmblock);
/*
  if ( cfg.cnn_cfg.scratch_size > 0 ) {
    size_t alloc_size = cfg.cnn_cfg.scratch_size;
    init_buf( (float*)(cfg.scratch), (alloc_size)/4, 0, 0 );
  }
*/
  at::Tensor output = input.new_empty(dim_size);

  void *input_pt = NULL, *output_pt = NULL, *filter_pt = NULL;

  libxsmm_tpp_convert_at_tensor_to_raw_helper(weight,         &filter_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(input,          &input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(output,         &output_pt);

#ifdef RECORD_FUNCTIONS_MACRO
  std::string label{"xsmm_conv_fwd_new"};
  label += "_";
  label += std::to_string(cfg.cnn_cfg.N);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.C);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.H);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.W);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.K);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.R);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.S);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.pad_h);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.pad_w);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.u);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.v);
#endif
//   auto start_time = Clock::now();

#ifdef DUMP_FORWARD_CONV
  FILE *f_input = NULL;
  f_input = fopen("input_conv_forward.txt", "wt");
  if (!f_input)
      printf("Failed to open f_input for writing\n");
  fprintf(f_input,"%d\n", cfg.cnn_cfg.N * cfg.cnn_cfg.C * cfg.cnn_cfg.H * cfg.cnn_cfg.W);
  for (int i = 0; i < cfg.cnn_cfg.N * cfg.cnn_cfg.C * cfg.cnn_cfg.H * cfg.cnn_cfg.W; i++)
    fprintf(f_input,"%6.6f\n", ((float*)input_pt)[i]);
  fclose(f_input);

  FILE *f_filter = NULL;
  f_filter = fopen("filter_conv_forward.txt", "wt");
  if (!f_filter)
      printf("Failed to open f_filter for writing\n");
  fprintf(f_filter,"%d\n", cfg.cnn_cfg.K * cfg.cnn_cfg.C * cfg.cnn_cfg.R * cfg.cnn_cfg.S);
  for (int i = 0; i < cfg.cnn_cfg.K * cfg.cnn_cfg.C * cfg.cnn_cfg.R * cfg.cnn_cfg.S; i++)
    fprintf(f_filter,"%6.6f\n", ((float*)filter_pt)[i]);
  fclose(f_filter);
#endif

  {
#ifdef RECORD_FUNCTIONS_MACRO
    //RECORD_FUNCTION("xsmm_conv_fwd_new", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
    RECORD_FUNCTION(label, std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      //CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );

//  int i = 0;
//  for (i = 0; i < 100; ++i)
//  {
      if (cfg.dtype == 0)
        libxsmm_dnn_conv_fwd_exec( cfg.cnn_cfg, (float*)filter_pt, (float*)input_pt, (float*)output_pt,
          NULL /*(float*)bias_pt*/, NULL /*(unsigned char*)relumask_pt*/, 0, tid, cfg.scratch );
      else
        libxsmm_dnn_conv_fwd_exec_bf16( cfg.cnn_cfg, (libxsmm_bfloat16*)filter_pt, (libxsmm_bfloat16*)input_pt, (libxsmm_bfloat16*)output_pt,
          NULL /*(float*)bias_pt*/, NULL /*(unsigned char*)relumask_pt*/, 0, tid, cfg.scratch );
//  } /* dummy i loop for standalone conv measurements */
    }
  }


#ifdef DUMP_FORWARD_CONV
/*
  FILE *f_output = NULL;
  f_output = fopen("output_conv_forward.txt", "wt");
  if (!f_output)
      printf("Failed to open f_output for writing\n");
  fprintf(f_output,"%d\n", cfg.cnn_cfg.N * cfg.cnn_cfg.K * cfg.cnn_cfg.ofhp * cfg.cnn_cfg.ofwp);
  for (int i = 0; i < cfg.cnn_cfg.N * cfg.cnn_cfg.K * cfg.cnn_cfg.ofhp * cfg.cnn_cfg.ofwp; i++)
    fprintf(f_output,"%6.6f\n", ((float*)output_pt)[i]);
  fclose(f_output);
*/
  naive_conv_t naive_param;

  my_eltwise_fuse my_fuse = LIBXSMM_DNN_ELTWISE_FUSE_NONE;

  naive_param.nImg = cfg.cnn_cfg.N;//nImg
  naive_param.nIfm = cfg.cnn_cfg.C;//nIfm;
  naive_param.nOfm = cfg.cnn_cfg.K;//nOfm;
  naive_param.ifhp = cfg.cnn_cfg.ifhp;
  naive_param.ifwp = cfg.cnn_cfg.ifwp;
  naive_param.ofhp = cfg.cnn_cfg.ofhp;
  naive_param.ofwp = cfg.cnn_cfg.ofwp;
  naive_param.ifh = cfg.cnn_cfg.H;//ifh;
  naive_param.ifw = cfg.cnn_cfg.W;//ifw;
  naive_param.ofh = cfg.cnn_cfg.ofh;
  naive_param.ofw = cfg.cnn_cfg.ofw;
  naive_param.pad_h = cfg.cnn_cfg.pad_h;
  naive_param.pad_w = cfg.cnn_cfg.pad_w;
  naive_param.pad_h_in = cfg.cnn_cfg.pad_h_in;
  naive_param.pad_w_in = cfg.cnn_cfg.pad_w_in;
  naive_param.pad_h_out = cfg.cnn_cfg.pad_h_out;
  naive_param.pad_w_out = cfg.cnn_cfg.pad_w_out;
  naive_param.kh = cfg.cnn_cfg.R;//kh;
  naive_param.kw = cfg.cnn_cfg.S;//kw;
  naive_param.stride_h = cfg.cnn_cfg.u;//stride_h;
  naive_param.stride_w = cfg.cnn_cfg.v;//stride_w;

  int nImg = cfg.cnn_cfg.N;
  int nIfm = cfg.cnn_cfg.C;
  int nOfm = cfg.cnn_cfg.K;
  int ifhp = cfg.cnn_cfg.ifhp;
  int ifwp = cfg.cnn_cfg.ifwp;
  int ofhp = cfg.cnn_cfg.ofhp;
  int ofwp = cfg.cnn_cfg.ofwp;
  int kh   = cfg.cnn_cfg.R;
  int kw   = cfg.cnn_cfg.S;

  float *naive_input, *naive_output, *naive_output_permuted, *naive_filter, *bias_libxsmm = NULL;

  printf("naive_param values: nImg nIfm nOfm ifhp, ifwp, ofhp, ofwp, ifh, ifw, ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out, kh, kw, stride_h, stride_w: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", 
                      naive_param.nImg, naive_param.nIfm, naive_param.nOfm, naive_param.ifhp, naive_param.ifwp, naive_param.ofhp, naive_param.ofwp, naive_param.ifh, naive_param.ifw, 
                        naive_param.ofh, naive_param.ofw, naive_param.pad_h, naive_param.pad_w, naive_param.pad_h_in, naive_param.pad_w_in, naive_param.pad_h_out, naive_param.pad_w_out,
                        naive_param.kh, naive_param.kw, naive_param.stride_h, naive_param.stride_w); fflush(0);
  printf("nImg nIfm nOfm ifhp ifwp ofhp ofwp kh kw = %d %d %d %d %d %d %d %d %d\n", nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, kh, kw);

  naive_input           = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_output          = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_permuted = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_filter          = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);

  //printf("block sizes from cfg.cnn_cfg: ifmblock = %d ofmblock = %d \n", cfg.cnn_cfg.ifmblock, cfg.cnn_cfg.ofmblock);

  tensor_copy_NCHWc_to_NCHW ((float*)input_pt,     naive_input,     nImg, nIfm, cfg.cnn_cfg.H, cfg.cnn_cfg.W, cfg.cnn_cfg.ifmblock);//N, C, H, W, bc);

  for (int i = 0; i < 10; i++)
        printf("input_pt[%d] = %15.15f naive_input[%d] = %15.15f \n", i, ((float*)input_pt)[i], i, naive_input[i]);

  //LIBXSMM_INLINE void tensor_copy_KCRSck_to_KCRS(float *src, float *dst, int K, int C, int R, int S, int bc, int bk)
  tensor_copy_KCRSck_to_KCRS((float*)filter_pt, naive_filter, nOfm, nIfm, kh, kw, cfg.cnn_cfg.ifmblock, cfg.cnn_cfg.ofmblock);

  //printf("Running naive_fused_conv_fp\n"); fflush(0);

  naive_fused_conv_fp(&naive_param, naive_input, naive_output, naive_filter, bias_libxsmm, (libxsmm_blasint)my_fuse);

  tensor_copy_NCHW_to_NCHWc       (naive_output, naive_output_permuted, nImg, nOfm, cfg.cnn_cfg.ofhp, cfg.cnn_cfg.ofwp, cfg.cnn_cfg.ofmblock);//N, C, H, W, bc);

  //printf("Called naive_fused_conv_fp\n"); fflush(0);

/*
  FILE *f_naive_output = fopen("naive_output_conv_forward.txt", "wt");
  fprintf(f_naive_output,"%d\n", cfg.cnn_cfg.N * cfg.cnn_cfg.K * cfg.cnn_cfg.ofhp * cfg.cnn_cfg.ofwp);
  for (int i = 0; i < cfg.cnn_cfg.N * cfg.cnn_cfg.K * cfg.cnn_cfg.ofhp * cfg.cnn_cfg.ofwp; i++)
    fprintf(f_naive_output,"%6.6f\n", ((float*)naive_output_permuted)[i]);
  fclose(f_naive_output);
*/
  //printf("Called matdiff\n"); fflush(0);

  libxsmm_matdiff_info norms;
  libxsmm_matdiff_clear(&norms);

  libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, nImg*nOfm*cfg.cnn_cfg.ofhp*cfg.cnn_cfg.ofwp, 1, naive_output_permuted, (float*)output_pt, 0, 0);
  printf("L1 reference  : %.25g\n", norms.l1_ref);
  printf("L1 test       : %.25g\n", norms.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms.l2_rel);
  printf("Linf abs.error: %.24f\n", norms.linf_abs);
  printf("Linf rel.error: %.24f\n", norms.linf_rel);
  printf("Check-norm    : %.24f\n", norms.normf_rel);

  if (nOfm == 256 && nIfm == 256 && kh == 3 && kw == 3 && ifhp == 14 && ofhp == 14) {
      printf("Want to stop and see\n");
  }

  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_output_permuted);
  libxsmm_free(naive_filter);
#endif

//   auto end_time = Clock::now();
//   std::cout << "Time difference:"
//      << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() * 1e-6 << " milliseconds" << std::endl;

  return output;

#endif /* if-else USE_OLD_HANDLE_GN */
}

std::vector<at::Tensor> conv_backward_new(pcl_cgbp_conv_config &cfg, torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight)
{
  /* printf("debug: calling conv_backward_new with tensor N H W C K R S: %d %d %d %d %d %d %d\n", cfg.cnn_cfg.N, cfg.cnn_cfg.H, cfg.cnn_cfg.W, cfg.cnn_cfg.C, cfg.cnn_cfg.K, cfg.cnn_cfg.R, cfg.cnn_cfg.S); */
#ifdef USE_OLD_HANDLE_CONV
  /* printf("debug in conv_backward_new but with an old implementation \n"); */
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_layer* libxsmm_handle = (libxsmm_dnn_layer*)(cfg.libxsmm_handle_);

  at::Tensor grad_input, grad_weight;
  grad_weight = at::empty(weight.sizes(), weight.options());

  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, grad_output, "Grad Output");
  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, weight, "Weight");
  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER, grad_weight, "Grad Weight");
  libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");

  if(!input.requires_grad())
  {
    grad_input = at::empty(input.sizes(), input.options());
    libxsmm_dnn_conv_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, grad_input, "Grad Input");
    RECORD_FUNCTION("xsmm_conv_bwdupd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid) );
    }
  }
  else
  {
    RECORD_FUNCTION("xsmm_conv_upd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid) );
    }
  }

  return {grad_input, grad_weight};
#else
  /* printf("debug: in conv_backward_new with a new implementation \n"); */
/*
  if ( cfg.cnn_cfg.scratch_size > 0 ) {
    size_t alloc_size = cfg.cnn_cfg.scratch_size;
    init_buf( (float*)(cfg.scratch), (alloc_size)/4, 0, 0 );
  }
*/
  at::Tensor grad_input, grad_weight;
  //grad_weight = at::empty(weight.sizes(), weight.options());
  grad_weight = at::zeros(weight.sizes(), weight.options());

//#define TIMING


#ifdef TIMING
  double t_start, time_c_bwd_d = 0.0, time_c_bwd_w = 0.0;
#endif

#ifdef TIMING
  t_start = getTime();
#endif

//#define SIMPLE_CONV_EXIT

#ifdef SIMPLE_CONV_EXIT
  #warning "SIMPLE_CONV_EXIT is active"

  std::cout << "grad_weight sizes in XsmmConv = " << grad_weight.sizes() << std::endl;

  if(input.requires_grad())
  {
    /* printf("debug: running bwd\n"); */
    grad_input = at::zeros(input.sizes(), input.options());
    std::cout << "grad_input sizes in XsmmConv = " << grad_input.sizes() << std::endl;
  }

  if(input.requires_grad())
    return {grad_input, grad_weight};
  else
    return {grad_weight};

#endif
  /*
  std::cout << "debug: input = " << input.sizes() << input.options() << std::endl;
  std::cout << "debug: weight = " << weight.sizes() << weight.options() << std::endl;
  std::cout << "debug: grad_output = " << grad_output.sizes() << grad_output.options() << std::endl;
  */

  void *input_pt = NULL, *grad_output_pt = NULL, *filter_pt = NULL, *grad_weight_pt = NULL, *grad_input_pt = NULL;

  libxsmm_tpp_convert_at_tensor_to_raw_helper(weight,         &filter_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(input,          &input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_output,    &grad_output_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_weight,    &grad_weight_pt);

#ifdef RECORD_FUNCTIONS_MACRO
  std::string label{"xsmm_conv_upd_new"};
  label += "_";
  label += std::to_string(cfg.cnn_cfg.N);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.C);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.H);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.W);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.K);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.R);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.S);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.pad_h);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.pad_w);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.u);
  label += "_";
  label += std::to_string(cfg.cnn_cfg.v);
#endif

  /* printf("debug: cfg.scratch = %p \n", cfg.scratch); */

  /*
  if (cfg.dtype == 0) {
    printf("debug: Printing here\n");
    for (int i = 0; i < 10; i++)
        printf("debug: grad_filter[%d] = %f \n", i, ((float*)grad_weight_pt)[i]);
    for (int i = 0; i < 10; i++)
        printf("debug: input[%d] = %f \n", i, ((float*)input_pt)[i]);
  }
  printf("debug: cfg pad_h pad_w ifhp ifwp = %d %d %d %d\n", cfg.cnn_cfg.pad_h, cfg.cnn_cfg.pad_w, cfg.cnn_cfg.ifhp, cfg.cnn_cfg.ifwp);
  */

  /* printf("debug:running upd\n"); */
#ifdef RECORD_FUNCTIONS_MACRO
  //RECORD_FUNCTION("xsmm_conv_upd_new", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
  RECORD_FUNCTION(label, std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    //CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid) );
    if (cfg.dtype == 0)
      libxsmm_dnn_conv_upd_exec( cfg.cnn_cfg, (const float*)input_pt, (const float*)grad_output_pt, (float*)grad_weight_pt,
        NULL, 0, tid, cfg.scratch );
    else
      libxsmm_dnn_conv_upd_exec_bf16( cfg.cnn_cfg, (const libxsmm_bfloat16*)input_pt, (const libxsmm_bfloat16*)grad_output_pt, (libxsmm_bfloat16*)grad_weight_pt,
        NULL, 0, tid, cfg.scratch );
  }

#ifdef TIMING
  time_c_bwd_w = getTime() - t_start;
  t_start = time_c_bwd_w + t_start;
#endif

  if(input.requires_grad())
  {
/*
    std::string label{"xsmm_conv_bwd_new"};
    label += "_";
    label += std::to_string(cfg.cnn_cfg.N);
    label += "_";
    label += std::to_string(cfg.cnn_cfg.C);
    label += "_";
    label += std::to_string(cfg.cnn_cfg.H);
    label += "_";
    label += std::to_string(cfg.cnn_cfg.W);
    label += "_";
    label += std::to_string(cfg.cnn_cfg.K);
    label += "_";
    label += std::to_string(cfg.cnn_cfg.R);
    label += "_";
    label += std::to_string(cfg.cnn_cfg.S);
    label += "_";
    label += std::to_string(cfg.cnn_cfg.pad_h);
    label += "_";
    label += std::to_string(cfg.cnn_cfg.pad_w);
    label += "_";
    label += std::to_string(cfg.cnn_cfg.u);
    label += "_";
    label += std::to_string(cfg.cnn_cfg.v);
*/

    /* printf("debug: running bwd\n"); */
    grad_input = at::empty(input.sizes(), input.options());
    libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_input, &grad_input_pt);

    //RECORD_FUNCTION("xsmm_conv_bwd_new", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
    //RECORD_FUNCTION(label, std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      //CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid) );
      if (cfg.dtype == 0)
        libxsmm_dnn_conv_bwd_exec( cfg.cnn_cfg, (const float*)filter_pt, NULL/*(float*)filtertr_libxsmm*/, (const float*)grad_output_pt, (float*)grad_input_pt,
            NULL/*(unsigned char*)relumask_pt*/, 0, tid, cfg.scratch );
      else
        libxsmm_dnn_conv_bwd_exec_bf16( cfg.cnn_cfg, (const libxsmm_bfloat16*)filter_pt, NULL/*(libxsmm_bfloat16*)filtertr_libxsmm*/, (const libxsmm_bfloat16*)grad_output_pt, (libxsmm_bfloat16*)grad_input_pt,
            NULL/*(unsigned char*)relumask_pt*/, 0, tid, cfg.scratch );
    }
  }

#ifdef TIMING
  time_c_bwd_d = getTime() - t_start;
  //t_start = t_c_bwd_d + t_start;
#endif

#if 0
  if(input.requires_grad()) {
    FILE *f_grad_input = fopen("bwd_conv_grad_input_new.txt", "wt");
    fprintf(f_grad_input,"%d\n", grad_input.numel());
    for (int i = 0; i < grad_input.numel(); i++)
      fprintf(f_grad_input,"%6.6f\n", ((float*)grad_input_pt)[i]);
    fclose(f_grad_input);
  }

  FILE *f_grad_weight = fopen("bwd_conv_grad_weight_new.txt", "wt");
  fprintf(f_grad_weight,"%d\n", grad_weight.numel());
  for (int i = 0; i < grad_weight.numel(); i++)
    fprintf(f_grad_weight,"%6.6f\n", ((float*)grad_weight_pt)[i]);
  fclose(f_grad_weight);

  exit(0);
#endif

  /*
  printf("input_pt       = %p \n", input_pt);
  printf("grad_output_pt = %p \n", grad_output_pt);
  printf("grad_filter    = %p \n", grad_weight_pt);
  if (cfg.dtype == 0)
    for (int i = 0; i < 10; i++)
        printf("grad_filter[%d] = %f \n", i, ((float*)grad_weight_pt)[i]);
  else
    for (int i = 0; i < 10; i++)
        printf("grad_filter[%d] = %hu \n", i, ((libxsmm_bfloat16*)grad_weight_pt)[i]);
  */

#if 0
for (int tensors = 0; tensors < 3; tensors++)
{
  int nbadp = 0, nchecked = 0;
  volatile void* p = NULL;
  volatile int nel = 0;

  if (p != NULL && nel != 0) {
//    if (rank == 0)
//      printf("doing the volatile-based check\n");
    for (int i = 0; i < nel; i++) {
      unsigned short * sval = reinterpret_cast<unsigned short*>(((libxsmm_bfloat16*)p + i));
      unsigned short checker = 0x7f80;

      unsigned short check_result = checker & (*sval);
      if (check_result == 0x7f80) {
//      if (rank == 0)
//        printf("nbad4 will be nonzero\n");
        nbadp++;
      }
      nchecked++;

    }

    {
      std::cout << "attempting to mprotect the pointer p " << reinterpret_cast<volatile void*>(p) << " up to ~ address " << reinterpret_cast<volatile void*>((libxsmm_bfloat16*)p + nel) << " size in bytes: " << nel*sizeof(libxsmm_bfloat16) << std::endl;
      //int len = nel * sizeof(libxsmm_bfloat16);
      //int len_pagealigned = ((4096-1)&len) ? ((len+4096) & ~(4096-1)):len;
      unsigned long long int len = - ((reinterpret_cast<std::uintptr_t>(p) & ~(4096-1)) + 4096) + (reinterpret_cast<std::uintptr_t>((libxsmm_bfloat16*)p + nel) & ~(4096-1));
      std::cout << "more exactly, we start on the next page boundary and do only len = " << len << " in bytes" << std::endl;
      int ierror = mprotect(reinterpret_cast<void*>((reinterpret_cast<std::uintptr_t>(p) & ~(4096-1)) + 4096 ), len, PROT_READ);
      if (ierror)
        std::cout << "ierror from mprotect = " << ierror << std::endl;
    }
  //if (rank == 0)
    printf("in XsmmConv conv_backward_new, tensor %d: p = %p nel = %d nbadp = %d nchecks = %d \n", tensors, p, nel, nbadp, nchecked);
  }
}
#endif

#ifdef TIMING
  if(input.requires_grad())
    printf("PERFDUMP,BD,resnetconv,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f\n",  (cfg.cnn_cfg.N), (cfg.cnn_cfg.N), (cfg.cnn_cfg.C), (cfg.cnn_cfg.K)  , (cfg.cnn_cfg.H), (cfg.cnn_cfg.W), (cfg.cnn_cfg.R), (cfg.cnn_cfg.S), (cfg.cnn_cfg.u), (cfg.cnn_cfg.pad_h), (cfg.cnn_cfg.pad_w),time_c_bwd_d, 1.0);//c1_mem_write_rfo_gb);
  printf("PERFDUMP,WU,resnetconv,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f\n",  (cfg.cnn_cfg.N), (cfg.cnn_cfg.N), (cfg.cnn_cfg.C), (cfg.cnn_cfg.K)  , (cfg.cnn_cfg.H), (cfg.cnn_cfg.W),  (cfg.cnn_cfg.R), (cfg.cnn_cfg.S), (cfg.cnn_cfg.u), (cfg.cnn_cfg.pad_h), (cfg.cnn_cfg.pad_w), time_c_bwd_w, 1.0);//c1_mem_write_rfo_gb);
#endif

  if(input.requires_grad())
    return {grad_input, grad_weight};
  else
    return {grad_weight};

//#undef TIMING

#endif /* if-else USE_OLD_HANDLE_GN */
}

#endif /* NEW_CONV */


#ifdef NEW_FC

typedef struct pcl_cgbp_fc_config {
  libxsmm_dnn_fc_fwd_config fwd_cfg;
  libxsmm_dnn_fc_bwd_config bwd_cfg;
  int              dtype;
  void*            scratch;
} pcl_cgbp_fc_config;

/* Returns a vector of size 2: {C_block, K_block} = {bc, bk} */
std::vector<int> fc_get_feature_map_blocks( int C, int K /*, datatype as an int flag? */ )
{
  std::vector<int> res;

  libxsmm_blasint bc = FC_BLOCK_SIZE; /* hardcoded for now */
  libxsmm_blasint bk = FC_BLOCK_SIZE; /* hardcoded for now */

  if (C % bc != 0)
    bc = C;
  if (K % bk != 0)
    bk = K;

  res.push_back(bc);
  res.push_back(bk);

  return res;
}

int fc_get_n_block( int N /*, datatype as an int flag? */ )
{
  libxsmm_blasint bn = FC_BLOCK_SIZE; /* hardcoded for now */

  if (N % bn != 0)
    bn = N;

  return bn;
}

pcl_cgbp_fc_config fc_setup_new(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint K, int fuse_type_int, int dtype_int, int bn_or_negative, int bc_or_negative, int bk_or_negative ) {
  pcl_cgbp_fc_config res;

  libxsmm_dnn_fc_eltw_fuse fuse_type_enum = LIBXSMM_DNN_FC_ELTW_FUSE_NONE;

  switch(fuse_type_int) {
    case 0:
      fuse_type_enum = LIBXSMM_DNN_FC_ELTW_FUSE_NONE;
      break;
    case 1:
      fuse_type_enum = LIBXSMM_DNN_FC_ELTW_FUSE_BIAS;
      break;
    case 2:
      fuse_type_enum = LIBXSMM_DNN_FC_ELTW_FUSE_RELU;
      break;
    case 4:
      fuse_type_enum = LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU;
      break;
    case 6:
      fuse_type_enum = LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK;
      break;
    case 7:
      fuse_type_enum = LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK;
      break;
    default:
      printf("Unrecognized fuse_type (int) = %d in fc_setup_new (assuming FUSE_NONE)\n", fuse_type_int);
      break;
  }

  libxsmm_blasint bn, bc, bk;

  if (bn_or_negative < 0) {
    libxsmm_blasint bn = fc_get_n_block( N /*, datatype as an int flag? */ );
  } else {
    bn = bn_or_negative;
  }

  if (bc_or_negative < 0 || bk_or_negative < 0) {
    std::vector<int> feature_map_blocks = fc_get_feature_map_blocks( C, K /*, datatype as an int flag? */ );
    libxsmm_blasint bc = feature_map_blocks[0];
    libxsmm_blasint bk = feature_map_blocks[1];
  } else {
    bc = bc_or_negative;
    bk = bk_or_negative;
  }

  //std::vector<int> feature_map_blocks = fc_get_feature_map_blocks( C, K /*, datatype as an int flag? */ );
  //libxsmm_blasint bc = feature_map_blocks[0];
  //libxsmm_blasint bk = feature_map_blocks[1];
  //libxsmm_blasint bn = fc_get_n_block( N /*, datatype as an int flag? */ );

  libxsmm_blasint threads = (libxsmm_blasint)omp_get_max_threads();

  libxsmm_datatype fc_dtype_in   = (dtype_int == 0 ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16);
  libxsmm_datatype fc_dtype_out  = fc_dtype_in;
  libxsmm_datatype fc_dtype_comp = LIBXSMM_DATATYPE_F32;

  res.dtype = dtype_int;

  /* printf("debug: N = %d C = %d K = %d bn = %d bc = %d bk = %d fuse_type = %d\n", N, C, K, bn, bc, bk, fuse_type_enum); */

  res.fwd_cfg = setup_libxsmm_dnn_fc_fwd(N, C, K, bn, bc, bk, threads, fuse_type_enum, fc_dtype_in, fc_dtype_out, fc_dtype_comp);
  res.bwd_cfg = setup_libxsmm_dnn_fc_bwd(N, C, K, bn, bc, bk, threads, fuse_type_enum, fc_dtype_in, fc_dtype_out, fc_dtype_comp);

  /* allocate and bind scratch */
  void *scratch = NULL;
  if ( res.fwd_cfg.scratch_size > 0 || res.bwd_cfg.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( res.fwd_cfg.scratch_size, res.bwd_cfg.scratch_size);
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    //init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
    zero_buf((float*)scratch, (alloc_size)/4);
  }

  res.scratch = scratch;

  return res;
}

void fc_setup_destroy_new(pcl_cgbp_fc_config cfg)
{
  if (cfg.scratch)
      libxsmm_free(cfg.scratch);
  cfg.scratch = NULL;

  destroy_libxsmm_dnn_fc_fwd(&cfg.fwd_cfg);
  destroy_libxsmm_dnn_fc_bwd(&cfg.bwd_cfg);

  return;
}

std::vector<at::Tensor> fc_forward_new(pcl_cgbp_fc_config &cfg, torch::Tensor input, torch::Tensor weight, torch::Tensor bias, std::vector<long> output_size)
{
  /* printf("debug: calling fc_forward_new with tensor N C K: %d %d %d\n", cfg.fwd_cfg.N, cfg.fwd_cfg.C, cfg.fwd_cfg.K); */
  /* printf("In fc_forward_new with a new implementation \n"); */
/*
  if ( cfg.fwd_cfg.scratch_size > 0 || cfg.bwd_cfg.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( cfg.fwd_cfg.scratch_size, cfg.bwd_cfg.scratch_size);
    init_buf( (float*)(cfg.scratch), (alloc_size)/4, 0, 0 );
  }
*/
  at::Tensor output    = input.new_empty(output_size);
  at::Tensor relumask  = at::empty(output_size, torch::TensorOptions().dtype(dt_map[7]));

  void *input_pt = NULL, *output_pt = NULL, *filter_pt = NULL, *bias_pt = NULL, *relumask_pt = NULL;

  libxsmm_tpp_convert_at_tensor_to_raw_helper(weight,         &filter_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(input,          &input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(output,         &output_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(bias,           &bias_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(relumask,       &relumask_pt);

#ifdef RECORD_FUNCTIONS_MACRO
  std::string label{"xsmm_fc_fwd_new"};
  label += "_";
  label += std::to_string(cfg.fwd_cfg.N);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.C);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.K);
#endif

  {
#ifdef RECORD_FUNCTIONS_MACRO
    //RECORD_FUNCTION("xsmm_fc_fwd_new", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
    RECORD_FUNCTION(label, std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();

      if (cfg.dtype == 0)
        libxsmm_dnn_fc_fwd_exec_f32( cfg.fwd_cfg, (float*)filter_pt, (float*)input_pt, (float*)output_pt,
            (float*)bias_pt, (unsigned char*)relumask_pt, 0, tid, cfg.scratch );
      else
        libxsmm_dnn_fc_fwd_exec_bf16( cfg.fwd_cfg, (libxsmm_bfloat16*)filter_pt, (libxsmm_bfloat16*)input_pt, (libxsmm_bfloat16*)output_pt,
            (libxsmm_bfloat16*)bias_pt, (unsigned char*)relumask_pt, 0, tid, cfg.scratch );
    }
  }

/*
{
    libxsmm_bfloat16* output_dbg = (libxsmm_bfloat16*)output_pt;//&LIBXSMM_VLA_ACCESS(2,  dbias, ofm1, 0, cfg.bk);
      for (int i = 0; i < output.numel(); i++) {
          float ftmp = 0.0;
          libxsmm_convert_bf16_f32(&(output_dbg[i]), &ftmp, 1);
          printf("output_dbg[%d] = %f as float\n", i, ftmp);
      }
}
*/
  return {output, relumask};
}

std::vector<at::Tensor> fc_backward_new(pcl_cgbp_fc_config &cfg, torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, torch::Tensor relumask, std::vector<long> bias_size)
{
  /* printf("debug: calling fc_backward_new with tensor N C K: %d %d %d\n", cfg.bwd_cfg.N, cfg.bwd_cfg..C, cfg.bwd_cfg.K); */
  /* printf("debug: in fc_backward_new with a new implementation \n"); */
/*
  if ( cfg.fwd_cfg.scratch_size > 0 || cfg.bwd_cfg.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( cfg.fwd_cfg.scratch_size, cfg.bwd_cfg.scratch_size);
    init_buf( (float*)(cfg.scratch), (alloc_size)/4, 0, 0 );
  }
*/
  at::Tensor grad_input, grad_weight, grad_bias;
  grad_weight = at::empty(weight.sizes(), weight.options());
  grad_input  = at::empty(input.sizes(),  input.options());
  grad_bias   = at::empty(bias_size, torch::TensorOptions().dtype(weight.dtype())); //at::empty(bias.sizes(),   bias.options());

  void *input_pt = NULL, *grad_output_pt = NULL, *filter_pt = NULL, *grad_weight_pt = NULL, *grad_input_pt = NULL, *grad_bias_pt = NULL, *relumask_pt = NULL;

  libxsmm_tpp_convert_at_tensor_to_raw_helper(weight,         &filter_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(input,          &input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_input,     &grad_input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_output,    &grad_output_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_weight,    &grad_weight_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_bias,      &grad_bias_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(relumask,       &relumask_pt);

  if(!input.requires_grad() || !weight.requires_grad())
  {
    printf("Smth is wrong, input and weight both should ask for gradient in fc bwd!\n");
    exit(-1);
  }

#ifdef RECORD_FUNCTIONS_MACRO
  std::string label{"xsmm_fc_bwdupd_new"};
  label += "_";
  label += std::to_string(cfg.bwd_cfg.N);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.C);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.K);
#endif

  /* printf("debug:running bwdupd\n"); */
#ifdef RECORD_FUNCTIONS_MACRO
  //RECORD_FUNCTION("xsmm_fc_bwdupd_new", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
  RECORD_FUNCTION(label, std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    if (cfg.dtype == 0)
      libxsmm_dnn_fc_bwd_exec_f32( cfg.bwd_cfg, (const float*)filter_pt, (float*)grad_input_pt,  (const float*)grad_output_pt, (float*)grad_weight_pt,
          (const float*)input_pt, (float*)grad_bias_pt, (const unsigned char*)relumask_pt, LIBXSMM_DNN_FC_PASS_BWD, 0, tid, cfg.scratch );
    else
      libxsmm_dnn_fc_bwd_exec_bf16( cfg.bwd_cfg, (const libxsmm_bfloat16*)filter_pt, (libxsmm_bfloat16*)grad_input_pt,  (const libxsmm_bfloat16*)grad_output_pt, (libxsmm_bfloat16*)grad_weight_pt,
          (const libxsmm_bfloat16*)input_pt, (libxsmm_bfloat16*)grad_bias_pt, (const unsigned char*)relumask_pt, LIBXSMM_DNN_FC_PASS_BWD, 0, tid, cfg.scratch );
  }

  return {grad_input, grad_weight, grad_bias};

}

#endif /* NEW_FC */

#ifdef NEW_POOLING

typedef struct pcl_cgbp_pooling_config {
  libxsmm_dnn_pooling_fwd_config fwd_cfg;
  libxsmm_dnn_pooling_bwd_config bwd_cfg;
  int                   pool_type; /* 0 for max; 1 for avg (mismatches the TPP enum!) */
  int                   dtype;     /* 0 for fp32, 1 for bf16 */
  void*                 scratch;
#ifdef USE_OLD_HANDLE_POOLING
  void*                 libxsmm_handle_;
#endif
} pcl_cgbp_pooling_config;

int pooling_get_c_block( int C /*, datatype as an int flag? */ )
{
  libxsmm_blasint bc = POOLING_BLOCK_SIZE; /* hardcoded for now */

  if (C % bc != 0)
    bc = C;

  return bc;
}

pcl_cgbp_pooling_config pooling_setup_new(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint R, libxsmm_blasint S,
                                    libxsmm_blasint padding, libxsmm_blasint stride, int pool_type_int, int dtype_int, int bc_or_negative ) {
  pcl_cgbp_pooling_config res;

  libxsmm_dnn_pooling_type pool_type_enum = LIBXSMM_DNN_POOLING_TYPE_MAX;

  switch(pool_type_int) {
    case 0:
      pool_type_enum = LIBXSMM_DNN_POOLING_TYPE_MAX;
      break;
    case 1:
      pool_type_enum = LIBXSMM_DNN_POOLING_TYPE_AVG;
      break;
    case 2:
      pool_type_enum = LIBXSMM_DNN_POOLING_TYPE_MAX_NOMASK;
      printf("Error: pool_type = LIBXSMM_DNN_POOLING_TYPE_MAX_NOMASK is not supported\n");
      exit(-1);
      break;
    default:
      printf("Unrecognized pool_type (int) = %d in pooling_setup_new\n", pool_type_int);
      exit(-1);
      break;
  }

  libxsmm_datatype pool_dtype_in   = (dtype_int == 0 ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16);
  libxsmm_datatype pool_dtype_out  = pool_dtype_in;
  libxsmm_datatype pool_dtype_comp = LIBXSMM_DATATYPE_F32;

  res.dtype = dtype_int;

  libxsmm_blasint bc;
  if (bc_or_negative < 0)
    bc = pooling_get_c_block(C);
  else
    bc = bc_or_negative;

  libxsmm_blasint threads = (libxsmm_blasint)omp_get_max_threads();

  res.fwd_cfg = setup_libxsmm_dnn_pooling_fwd(N, C, H, W, R, S, stride, stride, padding, padding, 0, 0, 0, 0 /*physical paddings */, bc, threads, pool_type_enum, pool_dtype_in, pool_dtype_out, pool_dtype_comp);
  res.bwd_cfg = setup_libxsmm_dnn_pooling_bwd(N, C, H, W, R, S, stride, stride, padding, padding, 0, 0, 0, 0 /*physical paddings */, bc, threads, pool_type_enum, pool_dtype_in, pool_dtype_out, pool_dtype_comp);

  /* allocate and bind scratch */
  void *scratch = NULL;
  if ( res.fwd_cfg.scratch_size > 0 || res.bwd_cfg.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( res.fwd_cfg.scratch_size, res.bwd_cfg.scratch_size);
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    //init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
    zero_buf((float*)scratch, (alloc_size)/4);
  }

  res.pool_type = pool_type_int;
  res.scratch   = scratch;

#ifdef USE_OLD_HANDLE_POOLING
  if (pool_type_enum == LIBXSMM_DNN_POOLING_TYPE_MAX)
    res.libxsmm_handle_ = max_pooling_create_handle(N, C, H, W, R, S, padding, stride, 1 /*int dtype*/);
  else if (pool_type_enum == LIBXSMM_DNN_POOLING_TYPE_AVG)
    res.libxsmm_handle_ = avg_pooling_create_handle(N, C, H, W, R, S, padding, stride, 1 /*int dtype*/);
  else
    res.libxsmm_handle_ = NULL;
#endif

  return res;
}

void pooling_setup_destroy_new(pcl_cgbp_pooling_config cfg)
{
  if (cfg.scratch)
    libxsmm_free(cfg.scratch);
  cfg.scratch = NULL;

  destroy_libxsmm_dnn_pooling_fwd(&cfg.fwd_cfg);
  destroy_libxsmm_dnn_pooling_bwd(&cfg.bwd_cfg);

#ifdef USE_OLD_HANDLE_POOLING
  if (cfg.pool_type == 0) /* max pooling */
    max_pooling_destroy_handle(cfg.libxsmm_handle_);
  else /* avg pooling */
    avg_pooling_destroy_handle(cfg.libxsmm_handle_);
  cfg.libxsmm_handle_ = NULL;
#endif

  return;
}

std::vector<at::Tensor> avg_pooling_forward_new(pcl_cgbp_pooling_config &cfg, torch::Tensor input, std::vector<long> output_size)
{
#ifdef USE_OLD_HANDLE_POOLING
//#if 0
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_pooling* libxsmm_handle = (libxsmm_dnn_pooling*)(cfg.libxsmm_handle_);

  libxsmm_dnn_avg_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");

  at::Tensor output;
  output = input.new_empty(output_size);
  libxsmm_dnn_avg_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, output, "Output");

  /* in fact, unused by the old implementation, yet present to have same API as the new one */
  at::Tensor relumask = at::empty(output_size, torch::TensorOptions().dtype(dt_map[7]));

  {
    RECORD_FUNCTION("xsmm_pooling_fwd_new", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
    }
  }
  return {output, relumask};
#else  /* USE_OLD_HANDLE_POOLING */
/*
  if ( cfg.fwd_cfg.scratch_size > 0 || cfg.bwd_cfg.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( cfg.fwd_cfg.scratch_size, cfg.bwd_cfg.scratch_size);
    init_buf( (float*)(cfg.scratch), (alloc_size)/4, 0, 0 );
  }
*/
  at::Tensor output, pool_mask;
  output    = input.new_empty(output_size);
  pool_mask = at::empty(output_size, torch::TensorOptions().dtype(dt_map[4])); /* int */

  void *input_pt = NULL, *output_pt = NULL, *pool_mask_pt = NULL;

  libxsmm_tpp_convert_at_tensor_to_raw_helper(input,     &input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(output,    &output_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(pool_mask, &pool_mask_pt);

#ifdef RECORD_FUNCTIONS_MACRO
  std::string label{"xsmm_avgpool_fwd_new"};
  label += "_";
  label += std::to_string(cfg.fwd_cfg.N);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.C);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.H);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.W);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.R);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.S);
#endif

  {
#ifdef RECORD_FUNCTIONS_MACRO
    //RECORD_FUNCTION("xsmm_avg_pooling_fwd_new", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
    RECORD_FUNCTION(label, std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      //CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
      if (cfg.dtype == 0)
        libxsmm_dnn_pooling_fwd_exec_f32( cfg.fwd_cfg, (const float*)input_pt, (float*)output_pt, (int*)pool_mask_pt,
                           0, tid, cfg.scratch );
      else
        libxsmm_dnn_pooling_fwd_exec_bf16( cfg.fwd_cfg, (const libxsmm_bfloat16*)input_pt, (libxsmm_bfloat16*)output_pt, (int*)pool_mask_pt,
                           0, tid, cfg.scratch );
    }
  }
  return {output, pool_mask};

#endif /* USE_OLD_HANDLE_POOLING */
}

at::Tensor avg_pooling_backward_new(pcl_cgbp_pooling_config &cfg, torch::Tensor grad_output, torch::Tensor pool_mask, std::vector<long> grad_in_size)
{
#ifdef USE_OLD_HANDLE_POOLING
//#if 0
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_pooling* libxsmm_handle = (libxsmm_dnn_pooling*)(cfg.libxsmm_handle_);

  libxsmm_dnn_avg_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, grad_output, "Grad Output");

  at::Tensor grad_input = at::empty(grad_in_size, grad_output.options());
  libxsmm_dnn_avg_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, grad_input, "Grad Input");

  {
    RECORD_FUNCTION("xsmm_pooling_bwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid) );
    }
  }
  return grad_input;
#else  /* USE_OLD_HANDLE_POOLING */
/*
  if ( cfg.fwd_cfg.scratch_size > 0 || cfg.bwd_cfg.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( cfg.fwd_cfg.scratch_size, cfg.bwd_cfg.scratch_size);
    init_buf( (float*)(cfg.scratch), (alloc_size)/4, 0, 0 );
  }
*/
  at::Tensor grad_input = at::empty(grad_in_size, grad_output.options());

  void *grad_output_pt = NULL, *grad_input_pt = NULL, *pool_mask_pt = NULL;

  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_input,  &grad_input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_output, &grad_output_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(pool_mask,   &pool_mask_pt);

#ifdef RECORD_FUNCTIONS_MACRO
  std::string label{"xsmm_avgpool_bwd_new"};
  label += "_";
  label += std::to_string(cfg.bwd_cfg.N);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.C);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.H);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.W);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.R);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.S);
#endif

  {
#ifdef RECORD_FUNCTIONS_MACRO
    //RECORD_FUNCTION("xsmm_avgpool_bwd_new", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
    RECORD_FUNCTION(label, std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      //CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid) );
      if (cfg.dtype == 0)
        libxsmm_dnn_pooling_bwd_exec_f32( cfg.bwd_cfg, (float*)grad_input_pt, (const float*)grad_output_pt, (const int*)pool_mask_pt,
                             0, tid, cfg.scratch );
      else
        libxsmm_dnn_pooling_bwd_exec_bf16( cfg.bwd_cfg, (libxsmm_bfloat16*)grad_input_pt, (const libxsmm_bfloat16*)grad_output_pt, (const int*)pool_mask_pt,
                             0, tid, cfg.scratch );
    }
  }
  return grad_input;

#endif /* USE_OLD_HANDLE_POOLING */
}


#ifdef POOLING_SCALAR_CODE
/*
typedef struct {
  int N;
  int C;
  int H;
  int W;
  int R;
  int S;
  int Cb;
  int bc;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int type;
} naive_pooling_t;
*/

LIBXSMM_INLINE void naive_pooling_fp_blocked_f32(naive_pooling_t* param, const float* input_ptr, float* output_ptr, int* mask_ptr)
{
  const int nImg = param->N;
  //const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int r = param->R;
  const int s = param->S;
  const int Cb = param->Cb;
  const int bc = param->bc;
  const int pad_h = param->pad_h;
  const int pad_w = param->pad_w;
  const int ofh = (ifh + 2*pad_h - r)/sh + 1;
  const int ofw = (ifw + 2*pad_w - s)/sw + 1;


  int img, lcb;

  LIBXSMM_VLA_DECL(5, const float, input,   input_ptr, Cb, ifh, ifw, bc);
  LIBXSMM_VLA_DECL(5,       int,   mask,     mask_ptr, Cb, ofh, ofw, bc);
  LIBXSMM_VLA_DECL(5,       float, output, output_ptr, Cb, ofh, ofw, bc);

#if defined(_OPENMP)
  float* tmp_buffer = (float*)malloc(sizeof(float)*ofh*ofw*bc*omp_get_max_threads());
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(lcb); //LIBXSMM_OMP_VAR(lbc);
# pragma omp parallel for private(img, lcb)
#else
  float* tmp_buffer = (float*)malloc(sizeof(float)*ofh*ofw);
#endif
  for (img = 0; img < nImg; img++) {
    for (lcb = 0; lcb < Cb; lcb++) {
#if defined(_OPENMP)
      float* lcl_buffer_ptr = tmp_buffer + (ofh*ofw*bc*omp_get_thread_num());
#else
      float* lcl_buffer_ptr = tmp_buffer;
#endif
      LIBXSMM_VLA_DECL(3, float, lcl_buffer, lcl_buffer_ptr, ofw, bc);
      int i, ho, wo, hi, wi, kh, kw, lbc;

      if (param->type == 0 ) {
        for ( i = 0; i < ofh*ofw*bc; i++ ) {
          lcl_buffer_ptr[i] = -FLT_MAX;
        }
      } else if (param->type == 1) {
        for ( i = 0; i < ofh*ofw*bc; i++ ) {
          lcl_buffer_ptr[i] = 0.0;
        }
      } else {
        /* should not happen */
      }

      for( ho = 0; ho < ofh; ho++ ) {
        hi = (ho * sh) - pad_h;
        for( wo = 0; wo < ofw; wo++ ) {
          wi = (wo * sw) - pad_w;
          for( kh = 0; kh < r; kh++ ) {
            if (hi+kh < 0 || hi+kh >= ifh) continue;
            for( kw = 0; kw < s; kw++ ) {
              if (wi+kw < 0 || wi+kw >= ifw) continue;

    for (lbc = 0; lbc < bc; lbc++) {

              if ( param->type == 0 ) {
                const int index = (hi+kh)*ifw + wi+kw;
                if ( LIBXSMM_VLA_ACCESS(5, input, img, lcb, hi+kh, wi+kw, lbc, Cb, ifh, ifw, bc) >= LIBXSMM_VLA_ACCESS(3, lcl_buffer, ho, wo, lbc, ofw, bc) ) {
                  LIBXSMM_VLA_ACCESS(3, lcl_buffer, ho, wo, lbc, ofw, bc) = LIBXSMM_VLA_ACCESS(5, input, img, lcb, hi+kh, wi+kw, lbc, Cb, ifh, ifw, bc);
                  LIBXSMM_VLA_ACCESS(5, mask, img, lcb, ho, wo, lbc, Cb, ofh, ofw, bc) = index;
                }
              } else if ( param->type == 1 ) {
                LIBXSMM_VLA_ACCESS(3, lcl_buffer, ho, wo, lbc, ofw, bc) += LIBXSMM_VLA_ACCESS(5, input, img, lcb, hi+kh, wi+kw, lbc, Cb, ifh, ifw, bc);
              } else {
                /* should not happen */
              }
    }
            }
          }
        }
      }

      if (param->type == 0 ) {
        for( ho = 0; ho < ofh; ho++ ) {
          for( wo = 0; wo < ofw; wo++ ) {
    for (lbc = 0; lbc < bc; lbc++) {
            LIBXSMM_VLA_ACCESS(5, output, img, lcb, ho, wo, lbc, Cb, ofh, ofw, bc) = LIBXSMM_VLA_ACCESS(3, lcl_buffer, ho, wo, lbc, ofw, bc);
    }
          }
        }
      } else if (param->type == 1) {
        for( ho = 0; ho < ofh; ho++ ) {
          for( wo = 0; wo < ofw; wo++ ) {
    for (lbc = 0; lbc < bc; lbc++) {
            LIBXSMM_VLA_ACCESS(5, output, img, lcb, ho, wo, lbc, Cb, ofh, ofw, bc) = LIBXSMM_VLA_ACCESS(3, lcl_buffer, ho, wo, lbc, ofw, bc) * (1.0f/(((float)r) * ((float)s)));
    }
          }
        }
      } else {
        /* should not happen */
      }
    }
  }

  free( tmp_buffer );
}

LIBXSMM_INLINE void naive_pooling_fp_blocked_bf16(naive_pooling_t* param, const libxsmm_bfloat16* input_ptr, libxsmm_bfloat16* output_ptr, int* mask_ptr)
{
  const int nImg = param->N;
  //const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int r = param->R;
  const int s = param->S;
  const int Cb = param->Cb;
  const int bc = param->bc;
  const int pad_h = param->pad_h;
  const int pad_w = param->pad_w;
  const int ofh = (ifh + 2*pad_h - r)/sh + 1;
  const int ofw = (ifw + 2*pad_w - s)/sw + 1;


  int img, lcb;

  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, input,   input_ptr, Cb, ifh, ifw, bc);
  LIBXSMM_VLA_DECL(5,       int,   mask,     mask_ptr, Cb, ofh, ofw, bc);
  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, output, output_ptr, Cb, ofh, ofw, bc);

#if defined(_OPENMP)
  float* tmp_buffer = (float*)malloc(sizeof(float)*ofh*ofw*bc*omp_get_max_threads());
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(lcb); //LIBXSMM_OMP_VAR(lbc);
# pragma omp parallel for private(img, lcb)
#else
  float* tmp_buffer = (float*)malloc(sizeof(float)*ofh*ofw);
#endif
  for (img = 0; img < nImg; img++) {
    for (lcb = 0; lcb < Cb; lcb++) {
#if defined(_OPENMP)
      float* lcl_buffer_ptr = tmp_buffer + (ofh*ofw*bc*omp_get_thread_num());
#else
      float* lcl_buffer_ptr = tmp_buffer;
#endif
      LIBXSMM_VLA_DECL(3, float, lcl_buffer, lcl_buffer_ptr, ofw, bc);
      int i, ho, wo, hi, wi, kh, kw, lbc;

      if (param->type == 0 ) {
        for ( i = 0; i < ofh*ofw*bc; i++ ) {
          lcl_buffer_ptr[i] = -FLT_MAX;
        }
      } else if (param->type == 1) {
        for ( i = 0; i < ofh*ofw*bc; i++ ) {
          lcl_buffer_ptr[i] = 0;
        }
      } else {
        /* should not happen */
      }

      for( ho = 0; ho < ofh; ho++ ) {
        hi = (ho * sh) - pad_h;
        for( wo = 0; wo < ofw; wo++ ) {
          wi = (wo * sw) - pad_w;
          for( kh = 0; kh < r; kh++ ) {
            if (hi+kh < 0 || hi+kh >= ifh) continue;
            for( kw = 0; kw < s; kw++ ) {
              if (wi+kw < 0 || wi+kw >= ifw) continue;

    for (lbc = 0; lbc < bc; lbc++) {

              if ( param->type == 0 ) {
                const int index = (hi+kh)*ifw + wi+kw;
                float tmp;
                libxsmm_convert_bf16_f32(&(LIBXSMM_VLA_ACCESS(5, input, img, lcb, hi+kh, wi+kw, lbc, Cb, ifh, ifw, bc)), &tmp, 1);
                //if ( LIBXSMM_VLA_ACCESS(5, input, img, lcb, hi+kh, wi+kw, lbc, Cb, ifh, ifw, bc) >= LIBXSMM_VLA_ACCESS(3, lcl_buffer, ho, wo, lbc, ofw, bc) ) {
                if ( tmp  >= LIBXSMM_VLA_ACCESS(3, lcl_buffer, ho, wo, lbc, ofw, bc) ) {
                  LIBXSMM_VLA_ACCESS(3, lcl_buffer, ho, wo, lbc, ofw, bc) = tmp;//LIBXSMM_VLA_ACCESS(5, input, img, lcb, hi+kh, wi+kw, lbc, Cb, ifh, ifw, bc);
                  LIBXSMM_VLA_ACCESS(5, mask, img, lcb, ho, wo, lbc, Cb, ofh, ofw, bc) = index;
                }
              } else if ( param->type == 1 ) {
        printf("param_type == 1 is not implemented in naive_pooling_fp_blocked_bf16\n");
        exit(-1);
                //LIBXSMM_VLA_ACCESS(3, lcl_buffer, ho, wo, lbc, ofw, bc) += LIBXSMM_VLA_ACCESS(5, input, img, lcb, hi+kh, wi+kw, lbc, Cb, ifh, ifw, bc);
              } else {
                /* should not happen */
              }
    }
            }
          }
        }
      }

      if (param->type == 0 ) {
        for( ho = 0; ho < ofh; ho++ ) {
          for( wo = 0; wo < ofw; wo++ ) {
    for (lbc = 0; lbc < bc; lbc++) {
            float tmp = LIBXSMM_VLA_ACCESS(3, lcl_buffer, ho, wo, lbc, ofw, bc);
            libxsmm_rne_convert_fp32_bf16(&tmp, &(LIBXSMM_VLA_ACCESS(5, output, img, lcb, ho, wo, lbc, Cb, ofh, ofw, bc)), 1);
            //LIBXSMM_VLA_ACCESS(5, output, img, lcb, ho, wo, lbc, Cb, ofh, ofw, bc) = LIBXSMM_VLA_ACCESS(3, lcl_buffer, ho, wo, lbc, ofw, bc);
    }
          }
        }
      } else if (param->type == 1) {
        printf("param_type == 1 is not implemented in naive_pooling_fp_blocked_bf16\n");
        exit(-1);
//        for( ho = 0; ho < ofh; ho++ ) {
//          for( wo = 0; wo < ofw; wo++ ) {
//    for (lbc = 0; lbc < bc; lbc++) {
//            LIBXSMM_VLA_ACCESS(5, output, img, lcb, ho, wo, lbc, Cb, ofh, ofw, bc) = LIBXSMM_VLA_ACCESS(3, lcl_buffer, ho, wo, lbc, ofw, bc) * (1.0f/(((float)r) * ((float)s)));
//    }
//          }
//        }
      } else {
        /* should not happen */
      }
    }
  }

  free( tmp_buffer );
}


LIBXSMM_INLINE void naive_pooling_bp_blocked_f32(naive_pooling_t* param, float* dinput_ptr, const float* doutput_ptr, const int* mask_ptr)
{
  const int nImg = param->N;
  //const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int r = param->R;
  const int s = param->S;
  const int Cb = param->Cb;
  const int bc = param->bc;
  const int pad_h = param->pad_h;
  const int pad_w = param->pad_w;
  const int ofh = (ifh + 2*pad_h - r)/sh + 1;
  const int ofw = (ifw + 2*pad_w - s)/sw + 1;

  int img, lcb;

  LIBXSMM_VLA_DECL(5,       float, dinput,   dinput_ptr, Cb, ifh, ifw, bc);
  LIBXSMM_VLA_DECL(5, const int  ,  mask,      mask_ptr, Cb, ofh, ofw, bc);
  LIBXSMM_VLA_DECL(5, const float, doutput, doutput_ptr, Cb, ofh, ofw, bc);

#if defined(_OPENMP)
  float* tmp_buffer = (float*)malloc(sizeof(float)*ifh*ifw*bc*omp_get_max_threads());
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(lcb);
# pragma omp parallel for private(img, lcb)
#else
  float* tmp_buffer = (float*)malloc(sizeof(float)*ofh*ofw);
#endif
  for (img = 0; img < nImg; img++) {
    for (lcb = 0; lcb < Cb; lcb++) {
#if defined(_OPENMP)
      float* lcl_buffer_ptr = tmp_buffer + (ifh*ifw*bc*omp_get_thread_num());
#else
      float* lcl_buffer_ptr = tmp_buffer;
#endif
      LIBXSMM_VLA_DECL(3, float, lcl_buffer, lcl_buffer_ptr, ifw, bc);
      int i, ho, wo, hi, wi, kh, kw, lbc;

      for ( i = 0; i < ifh*ifw*bc; i++ ) {
        lcl_buffer_ptr[i] = 0.0;
      }

      if (param->type == 0 ) {
        for( ho = 0; ho < ofh; ho++ ) {
          for( wo = 0; wo < ofw; wo++ ) {
    for (lbc = 0; lbc < bc; lbc++) {
            lcl_buffer_ptr[LIBXSMM_VLA_ACCESS(5, mask, img, lcb, ho, wo, lbc, Cb, ofh, ofw, bc)] += LIBXSMM_VLA_ACCESS(5, doutput, img, lcb, ho, wo, lbc, Cb, ofh, ofw, bc);
    }
          }
        }
      } else if ( param->type == 1 ) {
        printf("param_type == 1 is not implemented in naive_pooling_bp_blocked\n");
        exit(-1);
      } else {
        /* should not happen */
      }

      for( hi = 0; hi < ifh; hi++ ) {
        for( wi = 0; wi < ifw; wi++ ) {
    for (lbc = 0; lbc < bc; lbc++) {
          LIBXSMM_VLA_ACCESS(5, dinput, img, lcb, hi, wi, lbc, Cb, ifh, ifw, bc) = LIBXSMM_VLA_ACCESS(3, lcl_buffer, hi, wi, lbc, ifw, bc);
    }
        }
      }
    }
  }

  free( tmp_buffer );
}

LIBXSMM_INLINE void naive_pooling_bp_blocked_bf16(naive_pooling_t* param, libxsmm_bfloat16* dinput_ptr, const libxsmm_bfloat16* doutput_ptr, const int* mask_ptr)
{
  const int nImg = param->N;
  //const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int r = param->R;
  const int s = param->S;
  const int Cb = param->Cb;
  const int bc = param->bc;
  const int pad_h = param->pad_h;
  const int pad_w = param->pad_w;
  const int ofh = (ifh + 2*pad_h - r)/sh + 1;
  const int ofw = (ifw + 2*pad_w - s)/sw + 1;

  int img, lcb;

  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, dinput,   dinput_ptr, Cb, ifh, ifw, bc);
  LIBXSMM_VLA_DECL(5, const int  ,  mask,      mask_ptr, Cb, ofh, ofw, bc);
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, doutput, doutput_ptr, Cb, ofh, ofw, bc);

#if defined(_OPENMP)
  float* tmp_buffer = (float*)malloc(sizeof(float)*ifh*ifw*bc*omp_get_max_threads());
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(lcb);
# pragma omp parallel for private(img, lcb)
#else
  float* tmp_buffer = (float*)malloc(sizeof(float)*ofh*ofw);
#endif
  for (img = 0; img < nImg; img++) {
    for (lcb = 0; lcb < Cb; lcb++) {
#if defined(_OPENMP)
      float* lcl_buffer_ptr = tmp_buffer + (ifh*ifw*bc*omp_get_thread_num());
#else
      float* lcl_buffer_ptr = tmp_buffer;
#endif
      LIBXSMM_VLA_DECL(3, float, lcl_buffer, lcl_buffer_ptr, ifw, bc);
      int i, ho, wo, hi, wi, kh, kw, lbc;

      for ( i = 0; i < ifh*ifw*bc; i++ ) {
        lcl_buffer_ptr[i] = 0.0;
      }

      if (param->type == 0 ) {
        for( ho = 0; ho < ofh; ho++ ) {
          for( wo = 0; wo < ofw; wo++ ) {
    for (lbc = 0; lbc < bc; lbc++) {
            float tmp;
            libxsmm_convert_bf16_f32(&(LIBXSMM_VLA_ACCESS(5, doutput, img, lcb, ho, wo, lbc, Cb, ofh, ofw, bc)), &tmp, 1);
            lcl_buffer_ptr[LIBXSMM_VLA_ACCESS(5, mask, img, lcb, ho, wo, lbc, Cb, ofh, ofw, bc)] += tmp;//LIBXSMM_VLA_ACCESS(5, doutput, img, lcb, ho, wo, lbc, Cb, ofh, ofw, bc);
    }
          }
        }
      } else if ( param->type == 1 ) {
        printf("param_type == 1 is not implemented in naive_pooling_bp_blocked\n");
        exit(-1);
      } else {
        /* should not happen */
      }

      for( hi = 0; hi < ifh; hi++ ) {
        for( wi = 0; wi < ifw; wi++ ) {
    for (lbc = 0; lbc < bc; lbc++) {
          float tmp = LIBXSMM_VLA_ACCESS(3, lcl_buffer, hi, wi, lbc, ifw, bc);
          libxsmm_rne_convert_fp32_bf16(&tmp, &(LIBXSMM_VLA_ACCESS(5, dinput, img, lcb, hi, wi, lbc, Cb, ifh, ifw, bc)), 1);
          //LIBXSMM_VLA_ACCESS(5, dinput, img, lcb, hi, wi, lbc, Cb, ifh, ifw, bc) = LIBXSMM_VLA_ACCESS(3, lcl_buffer, hi, wi, lbc, ifw, bc);
    }
        }
      }
    }
  }

  free( tmp_buffer );
}

#endif // for #ifdef POOLING_SCALAR_CODE

std::vector<at::Tensor> max_pooling_forward_new(pcl_cgbp_pooling_config &cfg, torch::Tensor input, std::vector<long> output_size)
{
#ifdef USE_OLD_HANDLE_POOLING
//#if 0
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_pooling* libxsmm_handle = (libxsmm_dnn_pooling*)(cfg.libxsmm_handle_);

  libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");

  at::Tensor pool_mask = at::empty(output_size, torch::TensorOptions().dtype(at::kInt));
  libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_POOLING_MASK, pool_mask, "Pooling Mask");

  at::Tensor output = input.new_empty(output_size);

  libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, output, "Output");

  {
    RECORD_FUNCTION("xsmm_pooling_fwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
    }
  }
  return {output, pool_mask};
#else  /* USE_OLD_HANDLE_POOLING */
/*
  if ( cfg.fwd_cfg.scratch_size > 0 || cfg.bwd_cfg.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( cfg.fwd_cfg.scratch_size, cfg.bwd_cfg.scratch_size);
    init_buf( (float*)(cfg.scratch), (alloc_size)/4, 0, 0 );
  }
*/
  at::Tensor output, pool_mask;
  output    = input.new_empty(output_size);
  pool_mask = at::empty(output_size, torch::TensorOptions().dtype(dt_map[4]));

  void *input_pt = NULL, *output_pt = NULL, *pool_mask_pt = NULL;

  libxsmm_tpp_convert_at_tensor_to_raw_helper(input,     &input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(output,    &output_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(pool_mask, &pool_mask_pt);

#ifdef RECORD_FUNCTIONS_MACRO
  std::string label{"xsmm_maxpool_fwd_new"};
  label += "_";
  label += std::to_string(cfg.fwd_cfg.N);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.C);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.H);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.W);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.R);
  label += "_";
  label += std::to_string(cfg.fwd_cfg.S);
#endif

#ifdef POOLING_SCALAR_CODE
/*
typedef struct {
  int N;
  int C;
  int H;
  int W;
  int R;
  int S;
  int Cb;
  int bc;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int type;
} naive_pooling_t;
*/
  naive_pooling_t pool_handle;
  pool_handle.N = cfg.fwd_cfg.N;
  pool_handle.C = cfg.fwd_cfg.C;
  pool_handle.H = cfg.fwd_cfg.H;
  pool_handle.W = cfg.fwd_cfg.W;
  pool_handle.R = cfg.fwd_cfg.R;
  pool_handle.S = cfg.fwd_cfg.S;
  pool_handle.Cb = cfg.fwd_cfg.C / cfg.fwd_cfg.bc;
  pool_handle.bc = cfg.fwd_cfg.bc;
  pool_handle.pad_h = cfg.fwd_cfg.pad_h;
  pool_handle.pad_w = cfg.fwd_cfg.pad_w;
  pool_handle.stride_h = cfg.fwd_cfg.u;
  pool_handle.stride_w = cfg.fwd_cfg.v;
  pool_handle.type = (cfg.fwd_cfg.pool_type == LIBXSMM_DNN_POOLING_TYPE_MAX ? 0 : 1);

  if (cfg.dtype == 0) {
    naive_pooling_fp_blocked_f32(&pool_handle, (const float*)input_pt, (float*)output_pt, (int*)pool_mask_pt);
  } else {
    naive_pooling_fp_blocked_bf16(&pool_handle, (const libxsmm_bfloat16*)input_pt, (libxsmm_bfloat16*)output_pt, (int*)pool_mask_pt);
    //printf("POOLING_SCALAR_CODE fwd case has not been implemented for bf16\n");
    //exit(-1);
  }
#else
  {
#ifdef RECORD_FUNCTIONS_MACRO
    //RECORD_FUNCTION("xsmm_max_pooling_fwd_new", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
    RECORD_FUNCTION(label, std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      //CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
      if (cfg.dtype == 0)
        libxsmm_dnn_pooling_fwd_exec_f32( cfg.fwd_cfg, (const float*)input_pt, (float*)output_pt, (int*)pool_mask_pt,
                             0, tid, cfg.scratch );
      else
        libxsmm_dnn_pooling_fwd_exec_bf16( cfg.fwd_cfg, (const libxsmm_bfloat16*)input_pt, (libxsmm_bfloat16*)output_pt, (int*)pool_mask_pt,
                             0, tid, cfg.scratch );
    }
  }
#endif // for #ifdef-else POOLING_SCALAR_CODE
  return {output, pool_mask};

#endif /* USE_OLD_HANDLE_POOLING */
}

at::Tensor max_pooling_backward_new(pcl_cgbp_pooling_config &cfg, torch::Tensor grad_output, torch::Tensor pool_mask, std::vector<long> grad_in_size)
{
#ifdef USE_OLD_HANDLE_POOLING
//#if 0
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_pooling* libxsmm_handle = (libxsmm_dnn_pooling*)(cfg.libxsmm_handle_);

  libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, grad_output, "Grad Output");
  libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_POOLING_MASK, pool_mask, "Pooling Mask");

  at::Tensor grad_input = at::empty(grad_in_size, grad_output.options());
  libxsmm_dnn_max_pooling_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, grad_input, "Grad Input");

  {
    RECORD_FUNCTION("xsmm_pooling_bwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid) );
    }
  }
  return grad_input;
#else  /* USE_OLD_HANDLE_POOLING */
/*
  if ( cfg.fwd_cfg.scratch_size > 0 || cfg.bwd_cfg.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( cfg.fwd_cfg.scratch_size, cfg.bwd_cfg.scratch_size);
    init_buf( (float*)(cfg.scratch), (alloc_size)/4, 0, 0 );
  }
*/
  at::Tensor grad_input = at::empty(grad_in_size, grad_output.options());

  void *grad_output_pt = NULL, *grad_input_pt = NULL, *pool_mask_pt = NULL;

  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_input,  &grad_input_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(grad_output, &grad_output_pt);
  libxsmm_tpp_convert_at_tensor_to_raw_helper(pool_mask,   &pool_mask_pt);

#ifdef RECORD_FUNCTIONS_MACRO
  std::string label{"xsmm_maxpool_bwd_new"};
  label += "_";
  label += std::to_string(cfg.bwd_cfg.N);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.C);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.H);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.W);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.R);
  label += "_";
  label += std::to_string(cfg.bwd_cfg.S);
#endif

#ifdef POOLING_SCALAR_CODE
/*
typedef struct {
  int N;
  int C;
  int H;
  int W;
  int R;
  int S;
  int Cb;
  int bc;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int type;
} naive_pooling_t;
*/
  naive_pooling_t pool_handle;
  pool_handle.N = cfg.fwd_cfg.N;
  pool_handle.C = cfg.fwd_cfg.C;
  pool_handle.H = cfg.fwd_cfg.H;
  pool_handle.W = cfg.fwd_cfg.W;
  pool_handle.R = cfg.fwd_cfg.R;
  pool_handle.S = cfg.fwd_cfg.S;
  pool_handle.Cb = cfg.fwd_cfg.C / cfg.fwd_cfg.bc;
  pool_handle.bc = cfg.fwd_cfg.bc;
  pool_handle.pad_h = cfg.fwd_cfg.pad_h;
  pool_handle.pad_w = cfg.fwd_cfg.pad_w;
  pool_handle.stride_h = cfg.fwd_cfg.u;
  pool_handle.stride_w = cfg.fwd_cfg.v;
  pool_handle.type = (cfg.fwd_cfg.pool_type == LIBXSMM_DNN_POOLING_TYPE_MAX ? 0 : 1);

  //LIBXSMM_INLINE void naive_pooling_bp_blocked(naive_pooling_t* param, float* dinput_ptr, const float* doutput_ptr, const int* mask_ptr)
  if (cfg.dtype == 0) {
    naive_pooling_bp_blocked_f32(&pool_handle, (float*)grad_input_pt, (const float*)grad_output_pt, (const int*)pool_mask_pt);
  } else {
    naive_pooling_bp_blocked_bf16(&pool_handle, (libxsmm_bfloat16*)grad_input_pt, (const libxsmm_bfloat16*)grad_output_pt, (const int*)pool_mask_pt);
    //printf("POOLING_SCALAR_CODE bwd case has not been implemented for bf16\n");
    //exit(-1);
  }
#else

  {
#ifdef RECORD_FUNCTIONS_MACRO
    //RECORD_FUNCTION("xsmm_max_pooling_bwd_new", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
    RECORD_FUNCTION(label, std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      //CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid) );
      if (cfg.dtype == 0)
        libxsmm_dnn_pooling_bwd_exec_f32( cfg.bwd_cfg, (float*)grad_input_pt, (const float*)grad_output_pt, (const int*)pool_mask_pt,
                             0, tid, cfg.scratch );
      else
        libxsmm_dnn_pooling_bwd_exec_bf16( cfg.bwd_cfg, (libxsmm_bfloat16*)grad_input_pt, (const libxsmm_bfloat16*)grad_output_pt, (const int*)pool_mask_pt,
                             0, tid, cfg.scratch );
    }
  }
#endif // for #ifdef-else POOLING_SCALAR_CODE
  return grad_input;
#endif /* USE_OLD_HANDLE_POOLING */
}


#endif /* NEW_POOLING */

#ifdef NEW_BOTTLENECK

typedef struct pcl_cgbp_bottleneck_bn_config {
  libxsmm_blasint N;
  libxsmm_blasint inplanes;
  libxsmm_blasint H;
  libxsmm_blasint W;
  libxsmm_blasint planes;
  libxsmm_blasint stride;
  float bn_eps;
  float bn_momentum;
  libxsmm_blasint bn_track_running_stats_int;
  libxsmm_blasint expansion;
  libxsmm_blasint padding_3x3_type; /* 0 for logical, 1 for physical */
  libxsmm_blasint pad_size;         /* size of physical padding */
  libxsmm_blasint dtype_int;

  libxsmm_blasint has_residual_conv;

  libxsmm_blasint conv1_kernel_size;
  libxsmm_blasint conv1_stride     ;
  libxsmm_blasint conv1_padding    ;

  libxsmm_blasint conv2_kernel_size;
  libxsmm_blasint conv2_stride     ;
  libxsmm_blasint conv2_padding    ;

  libxsmm_blasint conv3_kernel_size;
  libxsmm_blasint conv3_stride     ;
  libxsmm_blasint conv3_padding    ;

  libxsmm_blasint conv4_kernel_size;
  libxsmm_blasint conv4_stride     ;
  libxsmm_blasint conv4_padding    ;

  libxsmm_blasint bn1_fuse_type;
  libxsmm_blasint bn2_fuse_type;
  libxsmm_blasint bn3_fuse_type;
  libxsmm_blasint bn4_fuse_type;

  pcl_cgbp_conv_config     conv1;
  pcl_cgbp_bn_config       bn1;
  pcl_cgbp_conv_config     conv2;
  pcl_cgbp_bn_config       bn2;
  pcl_cgbp_conv_config     conv3;
  pcl_cgbp_bn_config       bn3;
  pcl_cgbp_conv_config     conv4; /* (optional) */
  pcl_cgbp_bn_config       bn4;   /* (optional) */
} pcl_cgbp_bottleneck_bn_config;

typedef struct pcl_cgbp_bottleneck_gn_config {
  libxsmm_blasint N;
  libxsmm_blasint inplanes;
  libxsmm_blasint H;
  libxsmm_blasint W;
  libxsmm_blasint planes;
  libxsmm_blasint G;
  libxsmm_blasint stride;
  float norm_eps;
  libxsmm_blasint expansion;
  libxsmm_blasint padding_3x3_type; /* 0 for logical, 1 for physical */
  libxsmm_blasint pad_size;         /* size of physical padding */
  libxsmm_blasint dtype_int;

  libxsmm_blasint has_residual_conv;

  libxsmm_blasint conv1_kernel_size;
  libxsmm_blasint conv1_stride     ;
  libxsmm_blasint conv1_padding    ;

  libxsmm_blasint conv2_kernel_size;
  libxsmm_blasint conv2_stride     ;
  libxsmm_blasint conv2_padding    ;

  libxsmm_blasint conv3_kernel_size;
  libxsmm_blasint conv3_stride     ;
  libxsmm_blasint conv3_padding    ;

  libxsmm_blasint conv4_kernel_size;
  libxsmm_blasint conv4_stride     ;
  libxsmm_blasint conv4_padding    ;

  libxsmm_blasint bn1_fuse_type;
  libxsmm_blasint bn2_fuse_type;
  libxsmm_blasint bn3_fuse_type;
  libxsmm_blasint bn4_fuse_type;

  pcl_cgbp_conv_config     conv1;
  pcl_cgbp_gn_config       bn1;
  pcl_cgbp_conv_config     conv2;
  pcl_cgbp_gn_config       bn2;
  pcl_cgbp_conv_config     conv3;
  pcl_cgbp_gn_config       bn3;
  pcl_cgbp_conv_config     conv4; /* (optional) */
  pcl_cgbp_gn_config       bn4;   /* (optional) */
} pcl_cgbp_bottleneck_gn_config;

/*
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dtype=pcl_cgbp.global_dtype)
        if self.use_ref_bn != True:
            self.bn1 = nn.BatchNorm2d(planes, eps, relu=True, dtype=pcl_cgbp.global_dtype)
            #self.bn1  = nn.BatchNorm2d(planes, eps)
        else:
            self.bn1  = nn.BatchNorm2d(planes, eps)
            #self.bn1  = nn.BatchNorm2d(planes, eps, track_running_stats=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, dtype=pcl_cgbp.global_dtype)

        if self.use_ref_bn != True:
            self.bn2 = nn.BatchNorm2d(planes, eps, relu=True, dtype=pcl_cgbp.global_dtype)
            #self.bn2 = nn.BatchNorm2d(planes, eps)
        else:
            self.bn2  = nn.BatchNorm2d(planes, eps, dtype=pcl_cgbp.global_dtype)
            #self.bn2  = nn.BatchNorm2d(planes, eps, track_running_stats=False)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dtype=pcl_cgbp.global_dtype)

        if self.use_ref_bn != True:
            self.bn3 = nn.BatchNorm2d(planes * 4, eps, relu=True, eltwise=True, dtype=pcl_cgbp.global_dtype)
            #self.bn3  = nn.BatchNorm2d(planes * 4, eps)
        else:
            self.bn3  = nn.BatchNorm2d(planes * 4, eps, dtype=pcl_cgbp.global_dtype)
            #self.bn3  = nn.BatchNorm2d(planes * 4, eps, track_running_stats=False)
        self.downsample1 = downsample1 # this is the conv part of downsampling;
        self.downsample2 = downsample2 # this is the bn   part of downsampling;

        """
        downsample1 = None
        downsample2 = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample1 = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, dtype=pcl_cgbp.global_dtype)
            downsample2 = nn.BatchNorm2d(planes * block.expansion, self.eps, dtype=pcl_cgbp.global_dtype)
        """
*/

/*
    ------ conv1 -- bn1 -- conv2 (3x3) -- bn2 -- conv3 ---- bn3 (eltwise add) --
    \                                                     /
      \                                                 /
        ------------- conv4 -- bn4 / or id ------------
*/
pcl_cgbp_bottleneck_bn_config bottleneck_bn_setup_new(libxsmm_blasint N, libxsmm_blasint inplanes, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint planes, libxsmm_blasint stride,
                                                float eps, float bn_momentum, libxsmm_blasint bn_track_running_stats_int, libxsmm_blasint expansion,
                                                libxsmm_blasint padding_3x3_type, libxsmm_blasint dtype_int,
                                                int bc_conv1_or_neg, int bc_conv2_or_neg, int bc_conv3_or_neg, int bk_conv3_or_neg )
{
  pcl_cgbp_bottleneck_bn_config res;

  memset( &res, 0, sizeof(res));

  res.N = N;
  res.inplanes = inplanes;
  res.H = H;
  res.W = W;
  res.planes = planes;
  res.stride = stride;
  res.bn_eps = eps;
  res.bn_momentum = bn_momentum;
  res.bn_track_running_stats_int = bn_track_running_stats_int;
  res.expansion = expansion;
  res.padding_3x3_type = padding_3x3_type;
  res.dtype_int = dtype_int;

  res.has_residual_conv = ( (res.stride != 1 || res.inplanes != res.planes * expansion) ? 1 : 0);

  res.pad_size = 0;
  if (res.padding_3x3_type == 0)
    res.pad_size = 0;
  else  /* physical padding around 3x3 convolutions */
    res.pad_size = 1;

  res.conv1_kernel_size = 1;
  res.conv1_stride      = 1;
  res.conv1_padding     = 0;
  res.conv1 = conv_setup_new(res.N, res.inplanes, res.H, res.W, res.planes, res.conv1_kernel_size, res.conv1_kernel_size,
                             res.conv1_padding, res.conv1_padding, res.conv1_padding, res.conv1_padding, res.conv1_padding, res.conv1_padding,
                             res.conv1_stride, res.dtype_int, bc_conv1_or_neg, bc_conv2_or_neg);

  res.conv2_kernel_size = 3;
  res.conv2_stride      = res.stride;
  res.conv2_padding     = 1;
  if (res.padding_3x3_type == 0)
    res.conv2 = conv_setup_new(res.N, res.planes, res.H, res.W, res.planes, res.conv2_kernel_size, res.conv2_kernel_size,
                               res.conv2_padding, res.conv2_padding, 0, 0, 0, 0,
                               res.conv2_stride, res.dtype_int, bc_conv2_or_neg, bc_conv3_or_neg);
  else /* physical padding */
    res.conv2 = conv_setup_new(res.N, res.planes, res.H, res.W, res.planes, res.conv2_kernel_size, res.conv2_kernel_size,
                               res.conv2_padding, res.conv2_padding, res.conv2_padding, res.conv2_padding, res.conv2_padding, res.conv2_padding,
                               res.conv2_stride, res.dtype_int, bc_conv2_or_neg, bc_conv3_or_neg);

  libxsmm_blasint downsampled_H, downsampled_W;
  if (res.stride != 1) {
    downsampled_H = res.H / res.stride;
    downsampled_W = res.W / res.stride;
  } else {
    downsampled_H = res.H;
    downsampled_W = res.W;
  }

  res.conv3_kernel_size = 1;
  res.conv3_stride      = 1;
  res.conv3_padding     = 0;
  res.conv3 = conv_setup_new(res.N, res.planes, downsampled_H, downsampled_W, res.planes * res.expansion, res.conv3_kernel_size, res.conv3_kernel_size,
                             res.conv3_padding, res.conv3_padding, res.conv3_padding, res.conv3_padding, res.conv3_padding, res.conv3_padding,
                             res.conv3_stride, res.dtype_int, bc_conv3_or_neg, bk_conv3_or_neg);

  /* optionally output-padded batchnorm before 3x3 conv */
  res.bn1_fuse_type = 4;
  res.bn1   = bnorm_setup_new(res.N, res.planes, res.H, res.W, 0, 0, res.pad_size, res.pad_size, res.bn_eps, res.bn1_fuse_type, res.dtype_int, bc_conv2_or_neg);

  /* optionally input-padded batchnorm after 3x3 conv */
  res.bn2_fuse_type = 4;
  res.bn2   = bnorm_setup_new(res.N, res.planes, downsampled_H, downsampled_W, res.pad_size, res.pad_size, 0, 0, res.bn_eps, res.bn2_fuse_type, res.dtype_int, bc_conv3_or_neg);

  res.bn3_fuse_type = 5;
  res.bn3   = bnorm_setup_new(res.N, res.planes * res.expansion, downsampled_H, downsampled_W, 0, 0, 0, 0, res.bn_eps, res.bn3_fuse_type, res.dtype_int, bk_conv3_or_neg);

  if (res.has_residual_conv) {
    res.conv4_kernel_size = 1;
    res.conv4_stride      = res.stride;
    res.conv4_padding     = 0;
    res.conv4 = conv_setup_new(res.N, res.inplanes, res.H, res.W, res.planes * res.expansion, res.conv4_kernel_size, res.conv4_kernel_size,
                               res.conv4_padding, res.conv4_padding, res.conv4_padding, res.conv4_padding, res.conv4_padding, res.conv4_padding,
                               res.conv4_stride, res.dtype_int, bc_conv1_or_neg, bk_conv3_or_neg);

    res.bn4_fuse_type = 0;
    res.bn4   = bnorm_setup_new(res.N, res.planes * res.expansion, downsampled_H, downsampled_W, 0, 0, 0, 0, res.bn_eps, res.bn4_fuse_type, res.dtype_int, bk_conv3_or_neg);
  }

  return res;
}

pcl_cgbp_bottleneck_gn_config bottleneck_gn_setup_new(libxsmm_blasint N, libxsmm_blasint inplanes, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint planes, libxsmm_blasint G, libxsmm_blasint stride,
                                                float eps, libxsmm_blasint expansion,
                                                libxsmm_blasint padding_3x3_type, libxsmm_blasint dtype_int )
{
  pcl_cgbp_bottleneck_gn_config res;

  memset( &res, 0, sizeof(res));

  res.N = N;
  res.inplanes = inplanes;
  res.H = H;
  res.W = W;
  res.G = G;
  res.planes = planes;
  res.stride = stride;
  res.norm_eps  = eps;
  res.expansion = expansion;
  res.padding_3x3_type = padding_3x3_type;
  res.dtype_int = dtype_int;

  res.has_residual_conv = ( (res.stride != 1 || res.inplanes != res.planes * expansion) ? 1 : 0);

  res.pad_size = 0;
  if (res.padding_3x3_type == 0)
    res.pad_size = 0;
  else  /* physical padding around 3x3 convolutions */
    res.pad_size = 1;

  // FIXME: -1, -1 arguments are passed instead of block sizes as currently high-level itnerfaces don't have these parameters
  res.conv1_kernel_size = 1;
  res.conv1_stride      = 1;
  res.conv1_padding     = 0;
  res.conv1 = conv_setup_new(res.N, res.inplanes, res.H, res.W, res.planes, res.conv1_kernel_size, res.conv1_kernel_size,
                             res.conv1_padding, res.conv1_padding, res.conv1_padding, res.conv1_padding, res.conv1_padding, res.conv1_padding,
                             res.conv1_stride, res.dtype_int, -1, -1);

  res.conv2_kernel_size = 3;
  res.conv2_stride      = res.stride;
  res.conv2_padding     = 1;
  if (res.padding_3x3_type == 0)
    res.conv2 = conv_setup_new(res.N, res.planes, res.H, res.W, res.planes, res.conv2_kernel_size, res.conv2_kernel_size,
                               res.conv2_padding, res.conv2_padding, 0, 0, 0, 0,
                               res.conv2_stride, res.dtype_int, -1, -1);
  else /* physical padding */
    res.conv2 = conv_setup_new(res.N, res.planes, res.H, res.W, res.planes, res.conv2_kernel_size, res.conv2_kernel_size,
                               res.conv2_padding, res.conv2_padding, res.conv2_padding, res.conv2_padding, res.conv2_padding, res.conv2_padding,
                               res.conv2_stride, res.dtype_int, -1, -1);

  libxsmm_blasint downsampled_H, downsampled_W;
  if (res.stride != 1) {
    downsampled_H = res.H / res.stride;
    downsampled_W = res.W / res.stride;
  } else {
    downsampled_H = res.H;
    downsampled_W = res.W;
  }

  res.conv3_kernel_size = 1;
  res.conv3_stride      = 1;
  res.conv3_padding     = 0;
  res.conv3 = conv_setup_new(res.N, res.planes, downsampled_H, downsampled_W, res.planes * res.expansion, res.conv3_kernel_size, res.conv3_kernel_size,
                             res.conv3_padding, res.conv3_padding, res.conv3_padding, res.conv3_padding, res.conv3_padding, res.conv3_padding,
                             res.conv3_stride, res.dtype_int, -1, -1);

  /* optionally output-padded batchnorm before 3x3 conv */
  res.bn1_fuse_type = 4;
  res.bn1   = gnorm_setup_new(res.N, res.planes, res.H, res.W, res.G, 0, 0, res.pad_size, res.pad_size, res.norm_eps, res.bn1_fuse_type, res.dtype_int);

  /* optionally input-padded batchnorm after 3x3 conv */
  res.bn2_fuse_type = 4;
  res.bn2   = gnorm_setup_new(res.N, res.planes, downsampled_H, downsampled_W, res.G, res.pad_size, res.pad_size, 0, 0, res.norm_eps, res.bn2_fuse_type, res.dtype_int);

  res.bn3_fuse_type = 5;
  res.bn3   = gnorm_setup_new(res.N, res.planes * res.expansion, downsampled_H, downsampled_W, res.G, 0, 0, 0, 0, res.norm_eps, res.bn3_fuse_type, res.dtype_int);

  if (res.has_residual_conv) {
    res.conv4_kernel_size = 1;
    res.conv4_stride      = res.stride;
    res.conv4_padding     = 0;
    res.conv4 = conv_setup_new(res.N, res.inplanes, res.H, res.W, res.planes * res.expansion, res.conv4_kernel_size, res.conv4_kernel_size,
                               res.conv4_padding, res.conv4_padding, res.conv4_padding, res.conv4_padding, res.conv4_padding, res.conv4_padding,
                               res.conv4_stride, res.dtype_int, -1, -1);

    res.bn4_fuse_type = 0;
    res.bn4   = gnorm_setup_new(res.N, res.planes * res.expansion, downsampled_H, downsampled_W, res.G, 0, 0, 0, 0, res.norm_eps, res.bn4_fuse_type, res.dtype_int);
  }

  return res;
}


void bottleneck_bn_setup_destroy_new(pcl_cgbp_bottleneck_bn_config cfg)
{
  conv_setup_destroy_new(cfg.conv1);
  conv_setup_destroy_new(cfg.conv2);
  conv_setup_destroy_new(cfg.conv3);

  bnorm_setup_destroy_new(cfg.bn1);
  bnorm_setup_destroy_new(cfg.bn2);
  bnorm_setup_destroy_new(cfg.bn3);

  if (cfg.has_residual_conv) {
    conv_setup_destroy_new(cfg.conv4);
    bnorm_setup_destroy_new(cfg.bn4);
  }

  return;
}

void bottleneck_gn_setup_destroy_new(pcl_cgbp_bottleneck_gn_config cfg)
{
  conv_setup_destroy_new(cfg.conv1);
  conv_setup_destroy_new(cfg.conv2);
  conv_setup_destroy_new(cfg.conv3);

  gnorm_setup_destroy_new(cfg.bn1);
  gnorm_setup_destroy_new(cfg.bn2);
  gnorm_setup_destroy_new(cfg.bn3);

  if (cfg.has_residual_conv) {
    conv_setup_destroy_new(cfg.conv4);
    gnorm_setup_destroy_new(cfg.bn4);
  }

  return;
}


std::vector<at::Tensor> bottleneck_bn_forward_new(pcl_cgbp_bottleneck_bn_config &cfg, int bn_norm_type, std::vector<at::Tensor> inputs)
{
/*
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)

        if self.downsample1 is not None and self.downsample2 is not None:
            residual1 = self.downsample1(x)
            residual  = self.downsample2(residual1)

        out = self.bn3(out, residual)
        return out
*/

  auto input        = inputs[0];
  auto conv1_weight = inputs[1];
  auto conv2_weight = inputs[2];
  auto conv3_weight = inputs[3];
  auto conv4_weight = inputs[4];
  auto bn1_weight   = inputs[5];
  auto bn2_weight   = inputs[6];
  auto bn3_weight   = inputs[7];
  auto bn4_weight   = inputs[8];
  auto bn1_bias     = inputs[9];
  auto bn2_bias     = inputs[10];
  auto bn3_bias     = inputs[11];
  auto bn4_bias     = inputs[12];
  auto bn1_mean     = inputs[13];
  auto bn2_mean     = inputs[14];
  auto bn3_mean     = inputs[15];
  auto bn4_mean     = inputs[16];
  auto bn1_var      = inputs[17];
  auto bn2_var      = inputs[18];
  auto bn3_var      = inputs[19];
  auto bn4_var      = inputs[20];

  auto dummy_add    = at::Tensor();
  auto dummy_invstd = at::Tensor();
  auto dummy_return = at::Tensor();


#ifdef TIMING
  double time_c1 = 0.0, time_b1 = 0.0, time_c2 = 0.0, time_b2 = 0.0, time_c3 = 0.0, time_b3 = 0.0, time_c4 = 0.0, time_b4 = 0.0;
  double time_c1b1 = 0.0, time_c2b2 = 0.0, time_c3b3 = 0.0, time_c4b4 = 0.0;
  double time_b1stats = 0.0, time_b2stats = 0.0, time_b3stats = 0.0, time_b4stats = 0.0;
  double time_c1b1extra = 0.0, time_c2b2extra = 0.0, time_c3b3extra = 0.0, time_c4b4extra = 0.0;

  double t_start, t_conv_start, t_conv_end, t_bn_stats_end, t_bn_end, t_end;
#endif

#ifdef TIMING
  t_start = getTime();
#endif

  /*
  if (bn_norm_type == 1)
  {
      printf("bn_norm_type = 1\n");
  }
  */

  std::vector<long> conv1_output_size{cfg.conv1.cnn_cfg.N, cfg.conv1.cnn_cfg.blocksofm, cfg.conv1.cnn_cfg.ofhp, cfg.conv1.cnn_cfg.ofwp, cfg.conv1.cnn_cfg.ofmblock};
  auto conv1_out = conv_forward_new(cfg.conv1, input, conv1_weight, conv1_output_size);

#ifdef TIMING
  time_c1 = getTime() - t_start;
  t_start = time_c1 + t_start;
#endif

  std::vector<long> bn1_output_size{cfg.bn1.fwd_cfg.N, cfg.bn1.fwd_cfg.CP, cfg.bn1.fwd_cfg.H + 2 * cfg.pad_size, cfg.bn1.fwd_cfg.W + 2 * cfg.pad_size, cfg.bn1.fwd_cfg.bc};
  auto bn1_ret = bnorm_forward_new(cfg.bn1, conv1_out, dummy_add, bn1_weight, bn1_bias, bn1_mean, bn1_var, dummy_invstd, bn1_output_size, bn_norm_type);
  auto bn1_out = bn1_ret[0];
  auto bn1_relu_out = bn1_ret[1];

#ifdef TIMING
  time_b1 = getTime() - t_start;
  t_start = time_b1 + t_start;
#endif

  std::vector<long> conv2_output_size;
  conv2_output_size = {cfg.conv2.cnn_cfg.N, cfg.conv2.cnn_cfg.blocksofm, cfg.conv2.cnn_cfg.ofhp, cfg.conv2.cnn_cfg.ofwp, cfg.conv2.cnn_cfg.ofmblock};
  auto conv2_out = conv_forward_new(cfg.conv2, bn1_out, conv2_weight, conv2_output_size);

#ifdef TIMING
  time_c2 = getTime() - t_start;
  t_start = time_c2 + t_start;
#endif

  std::vector<long> bn2_output_size {cfg.bn2.fwd_cfg.N, cfg.bn2.fwd_cfg.CP, cfg.bn2.fwd_cfg.H, cfg.bn2.fwd_cfg.W, cfg.bn2.fwd_cfg.bc};
  auto bn2_ret   = bnorm_forward_new(cfg.bn2, conv2_out, dummy_add, bn2_weight, bn2_bias, bn2_mean, bn2_var, dummy_invstd, bn2_output_size, bn_norm_type);
  auto bn2_out = bn2_ret[0];
  auto bn2_relu_out = bn2_ret[1];

#ifdef TIMING
  time_b2 = getTime() - t_start;
  t_start = time_b2 + t_start;
#endif

  std::vector<long> conv3_output_size{cfg.conv3.cnn_cfg.N, cfg.conv3.cnn_cfg.blocksofm, cfg.conv3.cnn_cfg.ofhp, cfg.conv3.cnn_cfg.ofwp, cfg.conv3.cnn_cfg.ofmblock};
  auto conv3_out = conv_forward_new(cfg.conv3, bn2_out, conv3_weight, conv3_output_size);

#ifdef TIMING
  time_c3 = getTime() - t_start;
  t_start = time_c3 + t_start;
#endif

  at::Tensor conv4_out, residual, bn4_relu_out, bn3_out, bn3_relu_out;
  //std::vector<at::Tensor> bn4_ret;
  if (cfg.has_residual_conv) {
    std::vector<long> conv4_output_size{cfg.conv4.cnn_cfg.N, cfg.conv4.cnn_cfg.blocksofm, cfg.conv4.cnn_cfg.ofhp, cfg.conv4.cnn_cfg.ofwp, cfg.conv4.cnn_cfg.ofmblock};
    conv4_out = conv_forward_new(cfg.conv4, input, conv4_weight, conv4_output_size);

#ifdef TIMING
  time_c4 = getTime() - t_start;
  t_start = time_c4 + t_start;
#endif

    std::vector<long> bn4_output_size{cfg.bn4.fwd_cfg.N, cfg.bn4.fwd_cfg.CP, cfg.bn4.fwd_cfg.H, cfg.bn4.fwd_cfg.W, cfg.bn4.fwd_cfg.bc};
    auto bn4_ret  = bnorm_forward_new(cfg.bn4, conv4_out, dummy_add, bn4_weight, bn4_bias, bn4_mean, bn4_var, dummy_invstd, bn4_output_size, bn_norm_type);
    residual = bn4_ret[0];
    bn4_relu_out = bn4_ret[1];

#ifdef TIMING
  time_b4 = getTime() - t_start;
  t_start = time_b4 + t_start;
#endif
  } else {
    conv4_out    = dummy_return;
    residual     = input;
    bn4_relu_out = dummy_return;
  }

  std::vector<long> bn3_output_size{cfg.bn3.fwd_cfg.N, cfg.bn3.fwd_cfg.CP, cfg.bn3.fwd_cfg.H, cfg.bn3.fwd_cfg.W, cfg.bn3.fwd_cfg.bc};
  auto bn3_ret = bnorm_forward_new(cfg.bn3, conv3_out, residual, bn3_weight, bn3_bias, bn3_mean, bn3_var, dummy_invstd, bn3_output_size, bn_norm_type);
  bn3_out = bn3_ret[0];
  bn3_relu_out = bn3_ret[1];

#ifdef TIMING
  time_b3 = getTime() - t_start;
  //t_start = time_b3 + t_start;
#endif


#ifdef TIMING
  // FIXME: This is fp32 hardcoded hence all memory estimates are wrong for bf16
  typedef float T;
  //typedef bfloat16 T;

  int training = 1;

#if 1
        printf("perfdebug: checking for bottleneck in fwd with cfg C K H W stride: %d %d %d %d %d\n", cfg.inplanes, cfg.planes, cfg.H, cfg.W, cfg.stride);
        printf("activation size (in Mb, per core): (inp = c4_in -> c1 out = c2_in (stride) -> c2_out = c3_in -> c3_out = c4_out %f %f %f %f \n",
                                                                   (cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) / MB,
                                                                   (cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T) / MB,
                                                                   (cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) / MB,
                                                                   (4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) / MB );
        double c1_ab_size = ((cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.inplanes)*(cfg.planes)*1*1*sizeof(T)) / MB;
        double c2_ab_size = ((cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.planes)*(cfg.planes)*3*3*sizeof(T)) / MB;
        double c3_ab_size = ((cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.planes)*(4*cfg.planes)*1*1*sizeof(T)) / MB;
        double c4_ab_size = ((cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.inplanes)*(4*cfg.planes)*1*1*sizeof(T)) / MB;
        printf("conv input footprint (inp + weights) (in Mb, per core): %f %f %f %f (c1, c2, c3, c4)\n",
                                                                   c1_ab_size,
                                                                   c2_ab_size,
                                                                   c3_ab_size,
                                                                   c4_ab_size );

        //(2.0*(double)cfg.N*(double)cfg.C*(double)cfg.K*(double)cfg.R*(double)cfg.S*(double)cfg.ofh*(double)cfg.ofw)/(1000*1000*1000)
        double c1_gflop = (2.0*(double)cfg.N*(double)cfg.inplanes*(double)cfg.planes*(double)1*(double)1*(double)cfg.H*(double)cfg.W)/(1000*1000*1000);
        double c2_gflop = (2.0*(double)cfg.N*(double)cfg.planes*(double)cfg.planes*(double)3*(double)3*(double)(cfg.H/cfg.stride)*(double)(cfg.W/cfg.stride))/(1000*1000*1000);
        double c3_gflop = (2.0*(double)cfg.N*(double)cfg.planes*(double)4*cfg.planes*(double)1*(double)1*(double)(cfg.H/cfg.stride)*(double)(cfg.W/cfg.stride))/(1000*1000*1000);
        double c4_gflop = (2.0*(double)cfg.N*(double)cfg.inplanes*(double)4*cfg.planes*(double)1*(double)1*(double)(cfg.H/cfg.stride)*(double)(cfg.W/cfg.stride))/(1000*1000*1000);
        printf("theoretical total conv flop: %f %f %f %f (c1, c2, c3, c4)\n", c1_gflop, c2_gflop, c3_gflop, c4_gflop);

        double c1_mem_rfo_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (double)cfg.N*2*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.inplanes)*(cfg.planes)*1*1*sizeof(T)) / GB / time_c1;
        double c2_mem_rfo_gb = ((double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)   + (double)cfg.N*2*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.planes)*(cfg.planes)*3*3*sizeof(T)) / GB / time_c2;
        double c3_mem_rfo_gb = ((double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (double)cfg.N*2*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.planes)*(4*cfg.planes)*1*1*sizeof(T)) / GB / time_c3;
        double c4_mem_rfo_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (double)cfg.N*2*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.inplanes)*(4*cfg.planes)*1*1*sizeof(T)) / GB / time_c4;
        printf("theoretical conv flop/byte ratios: %f %f %f %f (c1, c2, c3, c4)\n", c1_gflop/c1_mem_rfo_gb, c2_gflop/c2_mem_rfo_gb, c3_gflop/c3_mem_rfo_gb, c4_gflop/c4_mem_rfo_gb);

        double c1_mem_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.inplanes)*(cfg.planes)*1*1*sizeof(T)) / GB;
        double c2_mem_gb = ((double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)   + (double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.planes)*(cfg.planes)*3*3*sizeof(T)) / GB;
        double c3_mem_gb = ((double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (double)cfg.N*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.planes)*(4*cfg.planes)*1*1*sizeof(T)) / GB;
        double c4_mem_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (double)cfg.N*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.inplanes)*(4*cfg.planes)*1*1*sizeof(T)) / GB;

        double c1_mem_act_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)) / GB;
        double c2_mem_act_gb = ((double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)   + (double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) ) / GB;
        double c3_mem_act_gb = ((double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (double)cfg.N*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) ) / GB;
        double c4_mem_act_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (double)cfg.N*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) ) / GB;

        double c1_mem_act_rfo_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + 2*(double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)) / GB;
        double c2_mem_act_rfo_gb = ((double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)   + 2*(double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) ) / GB;
        double c3_mem_act_rfo_gb = ((double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + 2*(double)cfg.N*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) ) / GB;
        double c4_mem_act_rfo_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + 2*(double)cfg.N*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) ) / GB;

        double c1_mem_write_rfo_gb = ((double)cfg.N*2*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)) / GB / time_c1;
        double c2_mem_write_rfo_gb = ((double)cfg.N*2*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T)) / GB / time_c2;
        double c3_mem_write_rfo_gb = ((double)cfg.N*2*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T)) / GB / time_c3;
        double c4_mem_write_rfo_gb = ((double)cfg.N*2*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T)) / GB / time_c4;

        printf("PERFDUMP,FP,resnetconv,%d,%d,%d,%d,%d,%d,1,1,1,0,0,%f,%f\n",  (cfg.N), (cfg.N), (cfg.inplanes), (cfg.planes)  , (cfg.H), (cfg.W), time_c1, c1_mem_write_rfo_gb);
        printf("PERFDUMP,FP,resnetconv,%d,%d,%d,%d,%d,%d,3,3,%d,1,1,%f,%f\n", (cfg.N), (cfg.N), (cfg.planes),   (cfg.planes)  , (cfg.H), (cfg.W), cfg.stride, time_c2, c2_mem_write_rfo_gb);
        printf("PERFDUMP,FP,resnetconv,%d,%d,%d,%d,%d,%d,1,1,1,0,0,%f,%f\n",  (cfg.N), (cfg.N), (cfg.planes),   (4*cfg.planes), (cfg.H / cfg.stride), (cfg.W / cfg.stride), time_c3, c3_mem_write_rfo_gb);
        if (cfg.has_residual_conv)
            printf("PERFDUMP,FP,resnetconv,%d,%d,%d,%d,%d,%d,1,1,%d,0,0,%f,%f\n", (cfg.N), (cfg.N), (cfg.inplanes), (4*cfg.planes), (cfg.H), (cfg.W), (cfg.stride), time_c4, c4_mem_write_rfo_gb);

        printf("PERFDUMP,FP,resnetbn,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%f,%f,%d,%d,%d\n", (cfg.N), (cfg.N), (cfg.planes)  , (cfg.planes)  , (cfg.H)             , (cfg.W)             , "na", "na", "na", (0), (1), time_b1, c1_ab_size, (1), (0), (training));
        printf("PERFDUMP,FP,resnetbn,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%f,%f,%d,%d,%d\n", (cfg.N), (cfg.N), (cfg.planes)  , (cfg.planes)  , (cfg.H / cfg.stride), (cfg.W / cfg.stride), "na", "na", "na", (1), (0), time_b2, c2_ab_size, (1), (0), (training));
        printf("PERFDUMP,FP,resnetbn,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%f,%f,%d,%d,%d\n", (cfg.N), (cfg.N), (4*cfg.planes), (4*cfg.planes), (cfg.H / cfg.stride), (cfg.W / cfg.stride), "na", "na", "na", (0), (0), time_b3, c3_ab_size, (1), (1), (training));
        if (cfg.has_residual_conv)
            printf("PERFDUMP,FP,resnetbn,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%f,1.0,%d,%d,%d\n", (cfg.N), (cfg.N), (4*cfg.planes), (4*cfg.planes), (cfg.H / cfg.stride), (cfg.W / cfg.stride)                 , "na", "na", "na", (0), (0), time_b4, c4_ab_size, (0), (0), (training));
#endif /* for #if 1/if 0 for PERFDUMP */
#endif /* for #ifdef TIMING */
  return {bn3_out, conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, residual, bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out};

}

std::vector<at::Tensor> bottleneck_gn_forward_new(pcl_cgbp_bottleneck_gn_config &cfg, std::vector<at::Tensor> inputs)
{
/*
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)

        if self.downsample1 is not None and self.downsample2 is not None:
            residual1 = self.downsample1(x)
            residual  = self.downsample2(residual1)

        out = self.bn3(out, residual)
        return out
*/

  auto input        = inputs[0];
  auto conv1_weight = inputs[1];
  auto conv2_weight = inputs[2];
  auto conv3_weight = inputs[3];
  auto conv4_weight = inputs[4];
  auto bn1_weight   = inputs[5];
  auto bn2_weight   = inputs[6];
  auto bn3_weight   = inputs[7];
  auto bn4_weight   = inputs[8];
  auto bn1_bias     = inputs[9];
  auto bn2_bias     = inputs[10];
  auto bn3_bias     = inputs[11];
  auto bn4_bias     = inputs[12];
  auto bn1_mean     = inputs[13];
  auto bn2_mean     = inputs[14];
  auto bn3_mean     = inputs[15];
  auto bn4_mean     = inputs[16];
  auto bn1_var      = inputs[17];
  auto bn2_var      = inputs[18];
  auto bn3_var      = inputs[19];
  auto bn4_var      = inputs[20];

  auto dummy_add    = at::Tensor();
  auto dummy_invstd = at::Tensor();
  auto dummy_return = at::Tensor();

  std::vector<long> conv1_output_size{cfg.conv1.cnn_cfg.N, cfg.conv1.cnn_cfg.blocksofm, cfg.conv1.cnn_cfg.ofhp, cfg.conv1.cnn_cfg.ofwp, cfg.conv1.cnn_cfg.ofmblock};
  auto conv1_out = conv_forward_new(cfg.conv1, input, conv1_weight, conv1_output_size);

  std::vector<long> bn1_output_size{cfg.bn1.fwd_cfg.N, cfg.bn1.fwd_cfg.CP, cfg.bn1.fwd_cfg.H + 2 * cfg.pad_size, cfg.bn1.fwd_cfg.W + 2 * cfg.pad_size, cfg.bn1.fwd_cfg.bc};
  auto bn1_ret = gnorm_forward_new(cfg.bn1, conv1_out, dummy_add, bn1_weight, bn1_bias, bn1_mean, bn1_var, dummy_invstd, bn1_output_size);
  auto bn1_out = bn1_ret[0];
  auto bn1_relu_out = bn1_ret[1];

  std::vector<long> conv2_output_size;
  conv2_output_size = {cfg.conv2.cnn_cfg.N, cfg.conv2.cnn_cfg.blocksofm, cfg.conv2.cnn_cfg.ofhp, cfg.conv2.cnn_cfg.ofwp, cfg.conv2.cnn_cfg.ofmblock};
  auto conv2_out = conv_forward_new(cfg.conv2, bn1_out, conv2_weight, conv2_output_size);

  std::vector<long> bn2_output_size {cfg.bn2.fwd_cfg.N, cfg.bn2.fwd_cfg.CP, cfg.bn2.fwd_cfg.H, cfg.bn2.fwd_cfg.W, cfg.bn2.fwd_cfg.bc};
  auto bn2_ret   = gnorm_forward_new(cfg.bn2, conv2_out, dummy_add, bn2_weight, bn2_bias, bn2_mean, bn2_var, dummy_invstd, bn2_output_size);
  auto bn2_out = bn2_ret[0];
  auto bn2_relu_out = bn2_ret[1];

  std::vector<long> conv3_output_size{cfg.conv3.cnn_cfg.N, cfg.conv3.cnn_cfg.blocksofm, cfg.conv3.cnn_cfg.ofhp, cfg.conv3.cnn_cfg.ofwp, cfg.conv3.cnn_cfg.ofmblock};
  auto conv3_out = conv_forward_new(cfg.conv3, bn2_out, conv3_weight, conv3_output_size);

  at::Tensor conv4_out, residual, bn4_relu_out, bn3_out, bn3_relu_out;
  if (cfg.has_residual_conv) {
    std::vector<long> conv4_output_size{cfg.conv4.cnn_cfg.N, cfg.conv4.cnn_cfg.blocksofm, cfg.conv4.cnn_cfg.ofhp, cfg.conv4.cnn_cfg.ofwp, cfg.conv4.cnn_cfg.ofmblock};
    conv4_out = conv_forward_new(cfg.conv4, input, conv4_weight, conv4_output_size);

    std::vector<long> bn4_output_size{cfg.bn4.fwd_cfg.N, cfg.bn4.fwd_cfg.CP, cfg.bn4.fwd_cfg.H, cfg.bn4.fwd_cfg.W, cfg.bn4.fwd_cfg.bc};
    auto bn4_ret  = gnorm_forward_new(cfg.bn4, conv4_out, dummy_add, bn4_weight, bn4_bias, bn4_mean, bn4_var, dummy_invstd, bn4_output_size);
    residual = bn4_ret[0];
    bn4_relu_out = bn4_ret[1];
  } else {
    conv4_out    = dummy_return;
    residual     = input;
    bn4_relu_out = dummy_return;
  }

  std::vector<long> bn3_output_size{cfg.bn3.fwd_cfg.N, cfg.bn3.fwd_cfg.CP, cfg.bn3.fwd_cfg.H, cfg.bn3.fwd_cfg.W, cfg.bn3.fwd_cfg.bc};
  auto bn3_ret = gnorm_forward_new(cfg.bn3, conv3_out, residual, bn3_weight, bn3_bias, bn3_mean, bn3_var, dummy_invstd, bn3_output_size);
  bn3_out = bn3_ret[0];
  bn3_relu_out = bn3_ret[1];

  return {bn3_out, conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, residual, bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out};

}


std::vector<at::Tensor> bottleneck_bn_backward_new(pcl_cgbp_bottleneck_bn_config &cfg, std::vector<at::Tensor> inputs)
{
  auto grad_output  = inputs[0];
  auto conv1_input  = inputs[1];
  auto conv1_weight = inputs[2];
  auto conv2_weight = inputs[3];
  auto conv3_weight = inputs[4];
  auto conv4_weight = inputs[5];
  auto bn1_weight   = inputs[6];
  auto bn2_weight   = inputs[7];
  auto bn3_weight   = inputs[8];
  auto bn4_weight   = inputs[9];
  auto bn1_bias     = inputs[10];
  auto bn2_bias     = inputs[11];
  auto bn3_bias     = inputs[12];
  auto bn4_bias     = inputs[13];
  auto bn1_mean     = inputs[14];
  auto bn2_mean     = inputs[15];
  auto bn3_mean     = inputs[16];
  auto bn4_mean     = inputs[17];
  auto bn1_var      = inputs[18];
  auto bn2_var      = inputs[19];
  auto bn3_var      = inputs[20];
  auto bn4_var      = inputs[21];

  auto conv1_out    = inputs[22];
  auto bn1_out      = inputs[23];
  auto conv2_out    = inputs[24];
  auto bn2_out      = inputs[25];
  auto conv3_out    = inputs[26];
  auto bn3_out      = inputs[27];
  auto conv4_out    = inputs[28];
  auto bn4_out      = inputs[29];

  auto bn1_relu_out = inputs[30];
  auto bn2_relu_out = inputs[31];
  auto bn3_relu_out = inputs[32];
  auto bn4_relu_out = inputs[33];

  auto dummy_output = at::Tensor();
  auto dummy_add    = at::Tensor();
  auto dummy_invstd = at::Tensor();
  auto dummy_return = at::Tensor();

#ifdef TIMING
  double time_c1 = 0.0, time_b1 = 0.0, time_c2 = 0.0, time_b2 = 0.0, time_c3 = 0.0, time_b3 = 0.0, time_c4 = 0.0, time_b4 = 0.0;
  double time_c1b1 = 0.0, time_c2b2 = 0.0, time_c3b3 = 0.0, time_c4b4 = 0.0;
  double time_b1stats = 0.0, time_b2stats = 0.0, time_b3stats = 0.0, time_b4stats = 0.0;
  double time_c1b1extra = 0.0, time_c2b2extra = 0.0, time_c3b3extra = 0.0, time_c4b4extra = 0.0;

  double t_start, t_conv_start, t_conv_end, t_bn_stats_end, t_bn_end, t_end;
#endif

#ifdef TIMING
  t_start = getTime();
#endif

  at::Tensor  conv1_grad_weight, conv2_grad_weight, conv3_grad_weight, conv4_grad_weight,
      bn1_grad_gamma, bn2_grad_gamma, bn3_grad_gamma, bn4_grad_gamma,
          bn1_grad_beta, bn2_grad_beta, bn3_grad_beta, bn4_grad_beta,
              conv1_grad_input, conv4_grad_input;

  auto residual           = bn4_out; // FIXME: Hopefully an alias and not a memory leak
  auto bn3_grad_ret       = bnorm_backward_new(cfg.bn3, grad_output /*grad_output*/, conv3_out, residual /*input_add*/, bn3_weight, dummy_output /*output*/, bn3_mean, bn3_var, dummy_invstd, bn3_relu_out);
  auto bn3_grad_input     = bn3_grad_ret[0];
  auto bn3_grad_input_add = bn3_grad_ret[1];
  bn3_grad_gamma          = bn3_grad_ret[2];
  bn3_grad_beta           = bn3_grad_ret[3];

#ifdef TIMING
  time_b3 = getTime() - t_start;
  t_start = time_b3 + t_start;
#endif

  bn2_out.requires_grad_(true);
  auto conv3_grad_ret   = conv_backward_new(cfg.conv3, bn3_grad_input /*grad_output*/, bn2_out, conv3_weight);
  auto conv3_grad_input = conv3_grad_ret[0];
  conv3_grad_weight     = conv3_grad_ret[1];

#ifdef TIMING
  time_c3 = getTime() - t_start;
  t_start = time_c3 + t_start;
#endif

  auto bn2_grad_ret   = bnorm_backward_new(cfg.bn2, conv3_grad_input /*grad_output*/, conv2_out, dummy_add, bn2_weight, dummy_output /*output*/, bn2_mean, bn2_var, dummy_invstd, bn2_relu_out);
  auto bn2_grad_input = bn2_grad_ret[0];
  bn2_grad_gamma      = bn2_grad_ret[2];
  bn2_grad_beta       = bn2_grad_ret[3];

#ifdef TIMING
  time_b2 = getTime() - t_start;
  t_start = time_b2 + t_start;
#endif

  bn1_out.requires_grad_(true);
  auto conv2_grad_ret   = conv_backward_new(cfg.conv2, bn2_grad_input /*grad_output*/, bn1_out, conv2_weight);
  auto conv2_grad_input = conv2_grad_ret[0];
  conv2_grad_weight     = conv2_grad_ret[1];

#ifdef TIMING
  time_c2 = getTime() - t_start;
  t_start = time_c2 + t_start;
#endif

  auto bn1_grad_ret   = bnorm_backward_new(cfg.bn1, conv2_grad_input /*grad_output*/, conv1_out, dummy_add, bn1_weight, dummy_output /*output*/, bn1_mean, bn1_var, dummy_invstd, bn1_relu_out);
  auto bn1_grad_input = bn1_grad_ret[0];
  bn1_grad_gamma      = bn1_grad_ret[2];
  bn1_grad_beta       = bn1_grad_ret[3];

#ifdef TIMING
  time_b1 = getTime() - t_start;
  t_start = time_b1 + t_start;
#endif

  conv1_input.requires_grad_(true);
  auto conv1_grad_ret = conv_backward_new(cfg.conv1, bn1_grad_input /*grad_output*/, conv1_input, conv1_weight);
  conv1_grad_input    = conv1_grad_ret[0];
  conv1_grad_weight   = conv1_grad_ret[1];

#ifdef TIMING
  time_c1 = getTime() - t_start;
  t_start = time_c1 + t_start;
#endif

  if (cfg.has_residual_conv) {

    auto bn4_grad_ret   = bnorm_backward_new(cfg.bn4, bn3_grad_input_add /*grad_output*/, conv4_out, dummy_add, bn4_weight, dummy_output /*output*/, bn4_mean, bn4_var, dummy_invstd, bn4_relu_out);
    auto bn4_grad_input = bn4_grad_ret[0];
    bn4_grad_gamma      = bn4_grad_ret[2];
    bn4_grad_beta       = bn4_grad_ret[3];

#ifdef TIMING
    time_b4 = getTime() - t_start;
    t_start = time_b4 + t_start;
#endif

    conv1_input.requires_grad_(true);
    auto conv4_grad_ret = conv_backward_new(cfg.conv4, bn4_grad_input /*grad_output*/, conv1_input, conv4_weight);
    conv4_grad_input    = conv4_grad_ret[0];
    conv4_grad_weight   = conv4_grad_ret[1];

#ifdef TIMING
    time_c4 = getTime() - t_start;
    //t_start = time_c4 + t_start;
#endif

  } else {
    conv4_grad_weight = dummy_return;
    bn4_grad_gamma    = dummy_return;
    bn4_grad_beta     = dummy_return;
    conv4_grad_input  = bn3_grad_input_add;
    conv4_grad_weight = dummy_return;
  }

#ifdef TIMING
  // FIXME: This is fp32 hardcoded hence all memory estimates are wrong for bf16
  typedef float T;
  //typedef bfloat16 T;

  int training = 1;

#if 1
        printf("perfdebug: checking for bottleneck in bwd with cfg C K H W stride: %d %d %d %d %d\n", cfg.inplanes, cfg.planes, cfg.H, cfg.W, cfg.stride);
/*
        printf("activation size (in Mb, per core): (inp = c4_in -> c1 out = c2_in (stride) -> c2_out = c3_in -> c3_out = c4_out %f %f %f %f \n",
                                                                   (cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) / MB,
                                                                   (cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T) / MB,
                                                                   (cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) / MB,
                                                                   (4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) / MB );
*/
        double c1_ab_size = ((cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.inplanes)*(cfg.planes)*1*1*sizeof(T)) / MB;
        double c2_ab_size = ((cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.planes)*(cfg.planes)*3*3*sizeof(T)) / MB;
        double c3_ab_size = ((cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.planes)*(4*cfg.planes)*1*1*sizeof(T)) / MB;
        double c4_ab_size = ((cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.inplanes)*(4*cfg.planes)*1*1*sizeof(T)) / MB;
/*
        printf("conv input footprint (inp + weights) (in Mb, per core): %f %f %f %f (c1, c2, c3, c4)\n",
                                                                   c1_ab_size,
                                                                   c2_ab_size,
                                                                   c3_ab_size,
                                                                   c4_ab_size );
*/
        //(2.0*(double)cfg.N*(double)cfg.C*(double)cfg.K*(double)cfg.R*(double)cfg.S*(double)cfg.ofh*(double)cfg.ofw)/(1000*1000*1000)
        double c1_gflop = 2.0*(2.0*(double)cfg.N*(double)cfg.inplanes*(double)cfg.planes*(double)1*(double)1*(double)cfg.H*(double)cfg.W)/(1000*1000*1000);
        double c2_gflop = 2.0*(2.0*(double)cfg.N*(double)cfg.planes*(double)cfg.planes*(double)3*(double)3*(double)(cfg.H/cfg.stride)*(double)(cfg.W/cfg.stride))/(1000*1000*1000);
        double c3_gflop = 2.0*(2.0*(double)cfg.N*(double)cfg.planes*(double)4*cfg.planes*(double)1*(double)1*(double)(cfg.H/cfg.stride)*(double)(cfg.W/cfg.stride))/(1000*1000*1000);
        double c4_gflop = 2.0*(2.0*(double)cfg.N*(double)cfg.inplanes*(double)4*cfg.planes*(double)1*(double)1*(double)(cfg.H/cfg.stride)*(double)(cfg.W/cfg.stride))/(1000*1000*1000);
        printf("theoretical total conv flop: %f %f %f %f (c1, c2, c3, c4)\n", c1_gflop, c2_gflop, c3_gflop, c4_gflop);

        double c1_mem_rfo_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (double)cfg.N*2*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.inplanes)*(cfg.planes)*1*1*sizeof(T)) / GB / time_c1;
        double c2_mem_rfo_gb = ((double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)   + (double)cfg.N*2*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.planes)*(cfg.planes)*3*3*sizeof(T)) / GB / time_c2;
        double c3_mem_rfo_gb = ((double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (double)cfg.N*2*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.planes)*(4*cfg.planes)*1*1*sizeof(T)) / GB / time_c3;
        double c4_mem_rfo_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (double)cfg.N*2*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.inplanes)*(4*cfg.planes)*1*1*sizeof(T)) / GB / time_c4;
        printf("theoretical conv flop/byte ratios: %f %f %f %f (c1, c2, c3, c4)\n", c1_gflop/c1_mem_rfo_gb, c2_gflop/c2_mem_rfo_gb, c3_gflop/c3_mem_rfo_gb, c4_gflop/c4_mem_rfo_gb);

        double c1_mem_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.inplanes)*(cfg.planes)*1*1*sizeof(T)) / GB;
        double c2_mem_gb = ((double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)   + (double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.planes)*(cfg.planes)*3*3*sizeof(T)) / GB;
        double c3_mem_gb = ((double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (double)cfg.N*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.planes)*(4*cfg.planes)*1*1*sizeof(T)) / GB;
        double c4_mem_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (double)cfg.N*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.inplanes)*(4*cfg.planes)*1*1*sizeof(T)) / GB;

        double c1_mem_act_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)) / GB;
        double c2_mem_act_gb = ((double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)   + (double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) ) / GB;
        double c3_mem_act_gb = ((double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (double)cfg.N*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) ) / GB;
        double c4_mem_act_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (double)cfg.N*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) ) / GB;

        double c1_mem_act_rfo_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + 2*(double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)) / GB;
        double c2_mem_act_rfo_gb = ((double)cfg.N*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)   + 2*(double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) ) / GB;
        double c3_mem_act_rfo_gb = ((double)cfg.N*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + 2*(double)cfg.N*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) ) / GB;
        double c4_mem_act_rfo_gb = ((double)cfg.N*(cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + 2*(double)cfg.N*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) ) / GB;

        double c1_mem_write_rfo_gb = ((double)cfg.N*2*(cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T)) / GB / time_c1;
        double c2_mem_write_rfo_gb = ((double)cfg.N*2*(cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T)) / GB / time_c2;
        double c3_mem_write_rfo_gb = ((double)cfg.N*2*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T)) / GB / time_c3;
        double c4_mem_write_rfo_gb = ((double)cfg.N*2*(4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T)) / GB / time_c4;

        printf("PERFDUMP,BP,resnetconv,%d,%d,%d,%d,%d,%d,1,1,1,0,0,%f,%f\n",  (cfg.N), (cfg.N), (cfg.inplanes), (cfg.planes)  , (cfg.H), (cfg.W), time_c1, c1_mem_write_rfo_gb);
        printf("PERFDUMP,BP,resnetconv,%d,%d,%d,%d,%d,%d,3,3,%d,1,1,%f,%f\n", (cfg.N), (cfg.N), (cfg.planes),   (cfg.planes)  , (cfg.H), (cfg.W), cfg.stride, time_c2, c2_mem_write_rfo_gb);
        printf("PERFDUMP,BP,resnetconv,%d,%d,%d,%d,%d,%d,1,1,1,0,0,%f,%f\n",  (cfg.N), (cfg.N), (cfg.planes),   (4*cfg.planes), (cfg.H / cfg.stride), (cfg.W / cfg.stride), time_c3, c3_mem_write_rfo_gb);
        if (cfg.has_residual_conv)
            printf("PERFDUMP,BP,resnetconv,%d,%d,%d,%d,%d,%d,1,1,%d,0,0,%f,%f\n", (cfg.N), (cfg.N), (cfg.inplanes), (4*cfg.planes), (cfg.H), (cfg.W), (cfg.stride), time_c4, c4_mem_write_rfo_gb);

        printf("PERFDUMP,BP,resnetbn,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%f,%f,%d,%d,%d\n", (cfg.N), (cfg.N), (cfg.planes)  , (cfg.planes)  , (cfg.H)             , (cfg.W)             , "na", "na", "na", (0), (1), time_b1, c1_ab_size, (1), (0), (training));
        printf("PERFDUMP,BP,resnetbn,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%f,%f,%d,%d,%d\n", (cfg.N), (cfg.N), (cfg.planes)  , (cfg.planes)  , (cfg.H / cfg.stride), (cfg.W / cfg.stride), "na", "na", "na", (1), (0), time_b2, c2_ab_size, (1), (0), (training));
        printf("PERFDUMP,BP,resnetbn,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%f,%f,%d,%d,%d\n", (cfg.N), (cfg.N), (4*cfg.planes), (4*cfg.planes), (cfg.H / cfg.stride), (cfg.W / cfg.stride), "na", "na", "na", (0), (0), time_b3, c3_ab_size, (1), (1), (training));
        if (cfg.has_residual_conv)
            printf("PERFDUMP,BP,resnetbn,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%f,1.0,%d,%d,%d\n", (cfg.N), (cfg.N), (4*cfg.planes), (4*cfg.planes), (cfg.H / cfg.stride), (cfg.W / cfg.stride)                 , "na", "na", "na", (0), (0), time_b4, c4_ab_size, (0), (0), (training));
#endif /* for #if 1/if 0 for PERFDUMP */
#endif /* for #ifdef TIMING */

  return {conv1_grad_weight, conv2_grad_weight, conv3_grad_weight, conv4_grad_weight,
          bn1_grad_gamma, bn2_grad_gamma, bn3_grad_gamma, bn4_grad_gamma,
          bn1_grad_beta, bn2_grad_beta, bn3_grad_beta, bn4_grad_beta,
          conv1_grad_input, conv4_grad_input};
}

std::vector<at::Tensor> bottleneck_gn_backward_new(pcl_cgbp_bottleneck_gn_config &cfg, std::vector<at::Tensor> inputs)
{
  auto grad_output  = inputs[0];
  auto conv1_input  = inputs[1];
  auto conv1_weight = inputs[2];
  auto conv2_weight = inputs[3];
  auto conv3_weight = inputs[4];
  auto conv4_weight = inputs[5];
  auto bn1_weight   = inputs[6];
  auto bn2_weight   = inputs[7];
  auto bn3_weight   = inputs[8];
  auto bn4_weight   = inputs[9];
  auto bn1_bias     = inputs[10];
  auto bn2_bias     = inputs[11];
  auto bn3_bias     = inputs[12];
  auto bn4_bias     = inputs[13];
  auto bn1_mean     = inputs[14];
  auto bn2_mean     = inputs[15];
  auto bn3_mean     = inputs[16];
  auto bn4_mean     = inputs[17];
  auto bn1_var      = inputs[18];
  auto bn2_var      = inputs[19];
  auto bn3_var      = inputs[20];
  auto bn4_var      = inputs[21];

  auto conv1_out    = inputs[22];
  auto bn1_out      = inputs[23];
  auto conv2_out    = inputs[24];
  auto bn2_out      = inputs[25];
  auto conv3_out    = inputs[26];
  auto bn3_out      = inputs[27];
  auto conv4_out    = inputs[28];
  auto bn4_out      = inputs[29];

  auto bn1_relu_out = inputs[30];
  auto bn2_relu_out = inputs[31];
  auto bn3_relu_out = inputs[32];
  auto bn4_relu_out = inputs[33];

  auto dummy_output = at::Tensor();
  auto dummy_add    = at::Tensor();
  auto dummy_invstd = at::Tensor();
  auto dummy_return = at::Tensor();

  at::Tensor  conv1_grad_weight, conv2_grad_weight, conv3_grad_weight, conv4_grad_weight,
      bn1_grad_gamma, bn2_grad_gamma, bn3_grad_gamma, bn4_grad_gamma,
          bn1_grad_beta, bn2_grad_beta, bn3_grad_beta, bn4_grad_beta,
              conv1_grad_input, conv4_grad_input;

  auto residual           = bn4_out; // FIXME: Hopefully an alias and not a memory leak
  auto bn3_grad_ret       = gnorm_backward_new(cfg.bn3, grad_output /*grad_output*/, conv3_out, residual /*input_add*/, bn3_weight, dummy_output /*output*/, bn3_mean, bn3_var, dummy_invstd, bn3_relu_out);
  auto bn3_grad_input     = bn3_grad_ret[0];
  auto bn3_grad_input_add = bn3_grad_ret[1];
  bn3_grad_gamma          = bn3_grad_ret[2];
  bn3_grad_beta           = bn3_grad_ret[3];

  bn2_out.requires_grad_(true);
  auto conv3_grad_ret   = conv_backward_new(cfg.conv3, bn3_grad_input /*grad_output*/, bn2_out, conv3_weight);
  auto conv3_grad_input = conv3_grad_ret[0];
  conv3_grad_weight     = conv3_grad_ret[1];

  auto bn2_grad_ret   = gnorm_backward_new(cfg.bn2, conv3_grad_input /*grad_output*/, conv2_out, dummy_add, bn2_weight, dummy_output /*output*/, bn2_mean, bn2_var, dummy_invstd, bn2_relu_out);
  auto bn2_grad_input = bn2_grad_ret[0];
  bn2_grad_gamma      = bn2_grad_ret[2];
  bn2_grad_beta       = bn2_grad_ret[3];

  bn1_out.requires_grad_(true);
  auto conv2_grad_ret   = conv_backward_new(cfg.conv2, bn2_grad_input /*grad_output*/, bn1_out, conv2_weight);
  auto conv2_grad_input = conv2_grad_ret[0];
  conv2_grad_weight     = conv2_grad_ret[1];

  auto bn1_grad_ret   = gnorm_backward_new(cfg.bn1, conv2_grad_input /*grad_output*/, conv1_out, dummy_add, bn1_weight, dummy_output /*output*/, bn1_mean, bn1_var, dummy_invstd, bn1_relu_out);
  auto bn1_grad_input = bn1_grad_ret[0];
  bn1_grad_gamma      = bn1_grad_ret[2];
  bn1_grad_beta       = bn1_grad_ret[3];

  conv1_input.requires_grad_(true);
  auto conv1_grad_ret = conv_backward_new(cfg.conv1, bn1_grad_input /*grad_output*/, conv1_input, conv1_weight);
  conv1_grad_input    = conv1_grad_ret[0];
  conv1_grad_weight   = conv1_grad_ret[1];

  if (cfg.has_residual_conv) {

    auto bn4_grad_ret   = gnorm_backward_new(cfg.bn4, bn3_grad_input_add /*grad_output*/, conv4_out, dummy_add, bn4_weight, dummy_output /*output*/, bn4_mean, bn4_var, dummy_invstd, bn4_relu_out);
    auto bn4_grad_input = bn4_grad_ret[0];
    bn4_grad_gamma      = bn4_grad_ret[2];
    bn4_grad_beta       = bn4_grad_ret[3];

    conv1_input.requires_grad_(true);
    auto conv4_grad_ret = conv_backward_new(cfg.conv4, bn4_grad_input /*grad_output*/, conv1_input, conv4_weight);
    conv4_grad_input    = conv4_grad_ret[0];
    conv4_grad_weight   = conv4_grad_ret[1];

  } else {
    conv4_grad_weight = dummy_return;
    bn4_grad_gamma    = dummy_return;
    bn4_grad_beta     = dummy_return;
    conv4_grad_input  = bn3_grad_input_add;
    conv4_grad_weight = dummy_return;
  }

  return {conv1_grad_weight, conv2_grad_weight, conv3_grad_weight, conv4_grad_weight,
          bn1_grad_gamma, bn2_grad_gamma, bn3_grad_gamma, bn4_grad_gamma,
          bn1_grad_beta, bn2_grad_beta, bn3_grad_beta, bn4_grad_beta,
          conv1_grad_input, conv4_grad_input};
}


#endif /* NEW_BOTTLENECK */

void
wait_for_debugger_local(int rank)
{
volatile int i = 0;
printf ( "pid %ld waiting for debugger \n", ( long ) getpid () ) ;
while ( i == 0) { /* change ’i’ in the debugger */ }
}

#ifndef PART_OF_EXTENSIONS
inline libxsmm_datatype convert_dtype_pt2xsmm(at::ScalarType dtype) {
  static const std::map<at::ScalarType, libxsmm_datatype> pt2xsmmDtypes = {
      {at::kDouble, LIBXSMM_DATATYPE_F64},
      {at::kFloat, LIBXSMM_DATATYPE_F32},
      {at::kHalf, LIBXSMM_DATATYPE_F16},
      {at::kBFloat16, LIBXSMM_DATATYPE_BF16},
      {at::kByte, LIBXSMM_DATATYPE_I8},
      {at::kChar, LIBXSMM_DATATYPE_I8},
      {at::kShort, LIBXSMM_DATATYPE_I16},
      {at::kInt, LIBXSMM_DATATYPE_I32},
      {at::kLong, LIBXSMM_DATATYPE_I64}};

  return pt2xsmmDtypes.at(dtype);
}

inline int xsmm_get_vnni_block_size(libxsmm_datatype dtype) {
  int bs = libxsmm_cpuid_dot_pack_factor(dtype);
  if (bs <= 0) {
    throw std::invalid_argument("Unsupported datatype");
  }
  return bs;
}

inline int cnn_vnni_block_size(at::ScalarType dtype) {
  auto xsmm_dtype = convert_dtype_pt2xsmm(dtype);
  return xsmm_get_vnni_block_size(xsmm_dtype);
}


int get_vnni_blocking(py::object dtype) {
  c10::ScalarType type = torch::python::detail::py_object_to_dtype(dtype);
  return cnn_vnni_block_size(type);
}
#endif

#ifndef PART_OF_EXTENSIONS
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
#else
REGISTER_SUBMODULE(_pcl_cgbp, m)
#endif
{
#ifdef NEW_BATCHNORM
  py::class_<pcl_cgbp_bn_config>(m, "pcl_cgbp_bn_config")
  .def(py::init<>())
  .def_readwrite("fwd_cfg",   &pcl_cgbp_bn_config::fwd_cfg)
  .def_readwrite("bwd_cfg",   &pcl_cgbp_bn_config::bwd_cfg)
  .def_readwrite("dtype",     &pcl_cgbp_bn_config::dtype)
  .def_readwrite("scratch",   &pcl_cgbp_bn_config::scratch)
#ifdef USE_OLD_HANDLE_BN
  .def_readwrite("eps", &pcl_cgbp_bn_config::eps)
  .def_readwrite("libxsmm_handle_", &pcl_cgbp_bn_config::libxsmm_handle_);
#else
  .def_readwrite("eps", &pcl_cgbp_bn_config::eps);
#endif
#endif /* NEW_BATCHNORM */

#ifdef NEW_GROUPNORM
  py::class_<pcl_cgbp_gn_config>(m, "pcl_cgbp_gn_config")
  .def(py::init<>())
  .def_readwrite("fwd_cfg", &pcl_cgbp_gn_config::fwd_cfg)
  .def_readwrite("bwd_cfg", &pcl_cgbp_gn_config::bwd_cfg)
  .def_readwrite("dtype",   &pcl_cgbp_gn_config::dtype)
  .def_readwrite("scratch", &pcl_cgbp_gn_config::scratch)
#ifdef USE_OLD_HANDLE_GN
  .def_readwrite("eps", &pcl_cgbp_gn_config::eps)
  .def_readwrite("libxsmm_handle_", &pcl_cgbp_gn_config::libxsmm_handle_);
#else
  .def_readwrite("eps", &pcl_cgbp_gn_config::eps);
#endif
#endif /* NEW_GROUPNORM */

#ifdef NEW_CONV
  py::class_<pcl_cgbp_conv_config>(m, "pcl_cgbp_conv_config")
  .def(py::init<>())
  .def_readwrite("cnn_cfg", &pcl_cgbp_conv_config::cnn_cfg)
  .def_readwrite("dtype", &pcl_cgbp_conv_config::dtype)
#ifdef USE_OLD_HANDLE_CONV
  .def_readwrite("scratch", &pcl_cgbp_conv_config::scratch)
  .def_readwrite("libxsmm_handle_", &pcl_cgbp_conv_config::libxsmm_handle_);
#else
  .def_readwrite("scratch", &pcl_cgbp_conv_config::scratch);
#endif
#endif /* NEW_CONV */

#ifdef NEW_FC
  py::class_<pcl_cgbp_fc_config>(m, "pcl_cgbp_fc_config")
  .def(py::init<>())
  .def_readwrite("fwd_cfg", &pcl_cgbp_fc_config::fwd_cfg)
  .def_readwrite("bwd_cfg", &pcl_cgbp_fc_config::bwd_cfg)
  .def_readwrite("dtype",   &pcl_cgbp_fc_config::dtype)
  .def_readwrite("scratch", &pcl_cgbp_fc_config::scratch);
#endif /* NEW_FC */

#ifdef NEW_POOLING
  py::class_<pcl_cgbp_pooling_config>(m, "pcl_cgbp_pooling_config")
  .def(py::init<>())
  .def_readwrite("fwd_cfg",   &pcl_cgbp_pooling_config::fwd_cfg)
  .def_readwrite("bwd_cfg",   &pcl_cgbp_pooling_config::bwd_cfg)
  .def_readwrite("pool_type", &pcl_cgbp_pooling_config::pool_type)
  .def_readwrite("dtype",     &pcl_cgbp_pooling_config::dtype)
#ifdef USE_OLD_HANDLE_POOLING
  .def_readwrite("scratch",   &pcl_cgbp_pooling_config::scratch)
  .def_readwrite("libxsmm_handle_", &pcl_cgbp_pooling_config::libxsmm_handle_);
#else
  .def_readwrite("scratch",   &pcl_cgbp_pooling_config::scratch);
#endif
#endif /* NEW_POOLING */

#ifdef NEW_BOTTLENECK
  py::class_<pcl_cgbp_bottleneck_bn_config>(m, "pcl_cgbp_bottleneck_bn_config")
  .def(py::init<>())
  .def_readwrite("dtype_int",   &pcl_cgbp_bottleneck_bn_config::dtype_int);
  py::class_<pcl_cgbp_bottleneck_gn_config>(m, "pcl_cgbp_bottleneck_gn_config")
  .def(py::init<>())
  .def_readwrite("dtype_int",   &pcl_cgbp_bottleneck_gn_config::dtype_int);
#endif /* NEW_BOTTLENECK */

#ifndef PART_OF_EXTENSIONS
  m.def("init_libxsmm",              &init_libxsmm,              "PCL LIBXSMM Init");
#endif

#ifdef OLD_LIBXSMM_HANDLES
  m.def("get_conv_tensor_layout",    &get_conv_tensor_layout,    "Pcl Conv tensor layout");
  m.def("get_gn_tensor_layout",      &get_gn_tensor_layout,      "Pcl GN Tensor layout");
  m.def("get_bn_tensor_layout",      &get_bn_tensor_layout,      "Pcl BN Tensor layout");
  m.def("get_pooling_tensor_layout", &get_pooling_tensor_layout, "Pcl Pool Tensor layout");

  m.def("conv_forward",        &conv_forward,        "Pcl libxsmm Conv forward");
  m.def("conv_backward",       &conv_backward,       "Pcl libxsmm Conv backward");
  m.def("conv_create_handle",  &conv_create_handle,  "Pcl libxsmm create Conv handle");
  m.def("conv_destroy_handle", &conv_destroy_handle, "Pcl libxsmm destroy Conv handle");

  m.def("gnorm_forward",                 &gnorm_forward,                 "Pcl libxsmm GN forward");
  m.def("gnorm_backward",                &gnorm_backward,                "Pcl libxsmm GN backward");
  m.def("fusedgroupnorm_create_handle",  &fusedgroupnorm_create_handle,  "Pcl libxsmm create GN handle");
  m.def("fusedgroupnorm_destroy_handle", &fusedgroupnorm_destroy_handle, "Pcl libxsmm destroy GN handle");
  m.def("bnorm_forward",                 &bnorm_forward,                 "Pcl libxsmm BN forward");
  m.def("bnorm_backward",                &bnorm_backward,                "Pcl libxsmm BN backward");
  m.def("fusedbatchnorm_create_handle",  &fusedbatchnorm_create_handle,  "Pcl libxsmm create BN handle");
  m.def("fusedbatchnorm_destroy_handle", &fusedbatchnorm_destroy_handle, "Pcl libxsmm destroy BN handle");
  m.def("avg_pooling_forward",           &avg_pooling_forward,        "Pcl libxsmm AvgPool forward");
  m.def("avg_pooling_backward",          &avg_pooling_backward,       "Pcl libxsmm AvgPool backward");
  m.def("avg_pooling_create_handle",     &avg_pooling_create_handle,  "Pcl libxsmm create AvgPool handle");
  m.def("avg_pooling_destroy_handle",    &avg_pooling_destroy_handle, "Pcl libxsmm destroy AvgPool handle");
  m.def("max_pooling_forward",           &max_pooling_forward,        "Pcl libxsmm MaxPool forward");
  m.def("max_pooling_backward",          &max_pooling_backward,       "Pcl libxsmm MaxPool backward");
  m.def("max_pooling_create_handle",     &max_pooling_create_handle,  "Pcl libxsmm create MaxPool handle");
  m.def("max_pooling_destroy_handle",    &max_pooling_destroy_handle, "Pcl libxsmm destroy MaxPool handle");
#endif /* OLD_LIBXSMM_HANDLES */

#ifdef NEW_CONV
  m.def("conv_forward_new",            &conv_forward_new,            "Pcl libxsmm CONV forward new");
  m.def("conv_backward_new",           &conv_backward_new,           "Pcl libxsmm CONV backward new");
  m.def("conv_setup_new",              &conv_setup_new,              "Pcl libxsmm CONV setup TPP");
  m.def("conv_setup_destroy_new",      &conv_setup_destroy_new,      "Pcl libxsmm CONV destroy TPP");
  #ifndef PART_OF_EXTENSIONS
  m.def("conv_get_feature_map_blocks", &conv_get_feature_map_blocks, "Pcl libxsmm CONV get feature map blocks for TPP");
  #endif
#endif
#ifdef NEW_GROUPNORM
  m.def("gnorm_forward_new",           &gnorm_forward_new,           "Pcl libxsmm GN forward new");
  m.def("gnorm_backward_new",          &gnorm_backward_new,          "Pcl libxsmm GN backward new");
  m.def("gnorm_setup_new",             &gnorm_setup_new,             "Pcl libxsmm GN setup TPP");
  m.def("gnorm_setup_destroy_new",     &gnorm_setup_destroy_new,     "Pcl libxsmm GN destroy TPP");
  m.def("gnorm_get_c_block",           &gnorm_get_c_block,           "Pcl libxsmm GN get c block for TPP");
#endif
#ifdef NEW_BATCHNORM
  m.def("bnorm_forward_new",           &bnorm_forward_new,           "Pcl libxsmm BN forward new");
  m.def("bnorm_backward_new",          &bnorm_backward_new,          "Pcl libxsmm BN backward new");
  m.def("bnorm_setup_new",             &bnorm_setup_new,             "Pcl libxsmm BN setup TPP");
  m.def("bnorm_setup_destroy_new",     &bnorm_setup_destroy_new,     "Pcl libxsmm BN destroy TPP");
  m.def("bnorm_get_c_block",           &bnorm_get_c_block,           "Pcl libxsmm BN get c block for TPP");
#endif
#ifdef NEW_POOLING
  m.def("pooling_setup_new",           &pooling_setup_new,           "Pcl libxsmm pooling setup TPP");
  m.def("pooling_setup_destroy_new",   &pooling_setup_destroy_new,   "Pcl libxsmm pooling destroy TPP");
  m.def("avg_pooling_forward_new",     &avg_pooling_forward_new,     "Pcl libxsmm AvgPool TPP forward");
  m.def("avg_pooling_backward_new",    &avg_pooling_backward_new,    "Pcl libxsmm AvgPool TPP backward");
  m.def("max_pooling_forward_new",     &max_pooling_forward_new,     "Pcl libxsmm MaxPool TPP forward");
  m.def("max_pooling_backward_new",    &max_pooling_backward_new,    "Pcl libxsmm MaxPool TPP backward");
  m.def("pooling_get_c_block",         &pooling_get_c_block,         "Pcl libxsmm Pool get c block for TPP");
#endif
#ifdef NEW_FC
  m.def("fc_forward_new",              &fc_forward_new,              "Pcl libxsmm FC forward new");
  m.def("fc_backward_new",             &fc_backward_new,             "Pcl libxsmm FC backward new");
  m.def("fc_setup_new",                &fc_setup_new,                "Pcl libxsmm FC setup TPP");
  m.def("fc_setup_destroy_new",        &fc_setup_destroy_new,        "Pcl libxsmm FC destroy TPP");
  m.def("fc_get_feature_map_blocks",   &fc_get_feature_map_blocks,   "Pcl libxsmm FC get feature map blocks for TPP");
  m.def("fc_get_n_block",              &fc_get_n_block,              "Pcl libxsmm FC get n block for TPP");
#endif
#ifdef NEW_BOTTLENECK
  m.def("bottleneck_bn_forward_new",              &bottleneck_bn_forward_new,              "Pcl libxsmm bottleneck bn forward new");
  m.def("bottleneck_bn_backward_new",             &bottleneck_bn_backward_new,             "Pcl libxsmm bottleneck bn backward new");
  m.def("bottleneck_bn_setup_new",                &bottleneck_bn_setup_new,                "Pcl libxsmm bottleneck bn setup TPP");
  m.def("bottleneck_bn_setup_destroy_new",        &bottleneck_bn_setup_destroy_new,        "Pcl libxsmm bottleneck bn destroy TPP");
  m.def("bottleneck_gn_forward_new",              &bottleneck_gn_forward_new,              "Pcl libxsmm bottleneck gn forward new");
  m.def("bottleneck_gn_backward_new",             &bottleneck_gn_backward_new,             "Pcl libxsmm bottleneck gn backward new");
  m.def("bottleneck_gn_setup_new",                &bottleneck_gn_setup_new,                "Pcl libxsmm bottleneck gn setup TPP");
  m.def("bottleneck_gn_setup_destroy_new",        &bottleneck_gn_setup_destroy_new,        "Pcl libxsmm bottleneck gn destroy TPP");
#endif
  m.def("wait_for_debugger_local", &wait_for_debugger_local, "wait_for_debugger_local");
#ifndef PART_OF_EXTENSIONS
  m.def("get_vnni_blocking", &get_vnni_blocking, "Wrapper around the LIBXSMM routine for getting the VNNI block size");
#endif
}
