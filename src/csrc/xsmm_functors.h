#ifndef _XSMM_FUNCTORS_H_
#define _XSMM_FUNCTORS_H_

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>
#else
#include <pytorch_extension_wrapper.h>
#endif
#include <bfloat8.h>
#include <string>
#include <unordered_map>


#define TPP_ASSERT(cond, x...) \
  do {                         \
    if (!(cond)) {             \
      printf(x);               \
      fflush(stdout);          \
      exit(1);                 \
    }                          \
  } while (0)
#define DECL_VLA_PTR(type, name, dims, ptr) type(*name) dims = (type(*) dims)ptr
#define ALIGNDOWN(N, A) ((N) & ~((A)-1))
namespace tpp {
typedef at::BFloat16 bfloat16;
typedef at::Half half;
typedef at::BFloat8 bfloat8;
inline float upconvert_to_float(float val) {
  return val;
}
inline float upconvert_to_float(bfloat16 val) {
  return (float)val;
}
inline float upconvert_to_float(half val) {
  return (float)val;
}
inline float upconvert_to_float(bfloat8 val) {
  return (float)val;
}
template <typename T>
inline libxsmm_datatype XsmmDtype();
template <>
inline libxsmm_datatype XsmmDtype<int64_t>() {
  return LIBXSMM_DATATYPE_I64;
}
template <>
inline libxsmm_datatype XsmmDtype<int32_t>() {
  return LIBXSMM_DATATYPE_I32;
}
template <>
inline libxsmm_datatype XsmmDtype<float>() {
  return LIBXSMM_DATATYPE_F32;
}
template <>
inline libxsmm_datatype XsmmDtype<bfloat16>() {
  return LIBXSMM_DATATYPE_BF16;
}
template <>
inline libxsmm_datatype XsmmDtype<half>() {
  return LIBXSMM_DATATYPE_F16;
}
template <>
inline libxsmm_datatype XsmmDtype<bfloat8>() {
  return LIBXSMM_DATATYPE_BF8;
}

#ifdef __AVX512F__
inline __m512 _mm512_loadu_ps_auto(float const* mem_addr) {
  return _mm512_loadu_ps(mem_addr);
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, float const* mem_addr) {
  return _mm512_maskz_loadu_ps(k, mem_addr);
}
inline void _mm512_storeu_ps_auto(float* mem_addr, __m512 a) {
  _mm512_storeu_ps(mem_addr, a);
}
inline void _mm512_mask_storeu_ps_auto(float* mem_addr, __mmask16 k, __m512 a) {
  _mm512_mask_storeu_ps(mem_addr, k, a);
}

inline __m512 _mm512_loadu_ps_auto(half const* mem_addr) {
  return _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, half const* mem_addr) {
  return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));
}
inline void _mm512_storeu_ps_auto(half* mem_addr, __m512 a) {
  _mm256_storeu_si256(
      (__m256i*)mem_addr,
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
inline void _mm512_mask_storeu_ps_auto(half* mem_addr, __mmask16 k, __m512 a) {
  _mm256_mask_storeu_epi16(
      (__m256i*)mem_addr,
      k,
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

inline __m512 _mm512_convert_bf_ps(__m256i a) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(a), 16));
}
inline __m256i _mm256_convert_ps_bf(__m512 a) {
  return _mm512_cvtepi32_epi16(
      _mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(a), 16));
}

inline __m512 _mm512_loadu_ps_auto(bfloat16 const* mem_addr) {
  return _mm512_convert_bf_ps(_mm256_loadu_si256((__m256i*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(
    __mmask16 k,
    bfloat16 const* mem_addr) {
  return _mm512_convert_bf_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));
}
inline void _mm512_storeu_ps_auto(bfloat16* mem_addr, __m512 a) {
  _mm256_storeu_si256((__m256i*)mem_addr, _mm256_convert_ps_bf(a));
}
inline void _mm512_mask_storeu_ps_auto(
    bfloat16* mem_addr,
    __mmask16 k,
    __m512 a) {
  _mm256_mask_storeu_epi16((__m256i*)mem_addr, k, _mm256_convert_ps_bf(a));
}

inline __m512 _mm512_split_loadu_ps(bfloat16 const* hi, bfloat16 const* lo) {
  auto yh = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)hi));
  auto yl = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)lo));
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
}
inline __m512 _mm512_maskz_split_loadu_ps(
    __mmask16 k,
    bfloat16 const* hi,
    bfloat16 const* lo) {
  auto yh = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)hi));
  auto yl = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)lo));
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
}
inline void _mm512_split_storeu_ps(bfloat16* hi, bfloat16* lo, __m512 a) {
  //_mm512_storeu_ps_auto(hi, a);
  _mm256_storeu_si256(
      (__m256i*)hi,
      _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_storeu_si256(
      (__m256i*)lo, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
inline void _mm512_mask_split_storeu_ps(
    bfloat16* hi,
    bfloat16* lo,
    __mmask16 k,
    __m512 a) {
  //_mm512_mask_storeu_ps_auto(hi, k, a);
  _mm256_mask_storeu_epi16(
      (__m256i*)hi,
      k,
      _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_mask_storeu_epi16(
      (__m256i*)lo, k, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
inline __m512 _mm512_convert_bf8_ps(__m128i a) {
  return _mm512_cvtph_ps(_mm256_slli_epi16(_mm256_cvtepi8_epi16(a), 8));
}
inline __m128i _mm_convert_ps_bf8(__m512 a) {
  return _mm256_cvtepi16_epi8(_mm256_srai_epi16(
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), 8));
}

inline __m512 _mm512_loadu_ps_auto(bfloat8 const* mem_addr) {
  return _mm512_convert_bf8_ps(_mm_loadu_si128((__m128i const*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, bfloat8 const* mem_addr) {
  return _mm512_convert_bf8_ps(
      _mm_maskz_loadu_epi8(k, (__m128i const*)mem_addr));
}
inline void _mm512_storeu_ps_auto(bfloat8* mem_addr, __m512 a) {
  _mm_storeu_si128((__m128i*)mem_addr, _mm_convert_ps_bf8(a));
}
inline void _mm512_mask_storeu_ps_auto(
    bfloat8* mem_addr,
    __mmask16 k,
    __m512 a) {
  _mm_mask_storeu_epi8((__m128i*)mem_addr, k, _mm_convert_ps_bf8(a));
}
#endif

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

inline int get_vnni_block_size(at::ScalarType dtype) {
  auto xsmm_dtype = convert_dtype_pt2xsmm(dtype);
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

inline int get_vnni_block_size(caffe2::TypeMeta dtype_) {
  at::ScalarType dtype = dtype_.toScalarType();
  auto xsmm_dtype = convert_dtype_pt2xsmm(dtype);
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

template <typename T>
inline int get_vnni_block_size() {
  auto xsmm_dtype = XsmmDtype<T>();
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

inline void debug_print_eqn_tree(libxsmm_blasint eqn_no) {
  if (false) {
    libxsmm_matrix_eqn_tree_print(eqn_no);
    libxsmm_matrix_eqn_rpn_print(eqn_no);
  }
}

inline int meqn_push_arg(
    const libxsmm_blasint idx,
    const libxsmm_blasint m,
    const libxsmm_blasint n,
    const libxsmm_blasint ld,
    const libxsmm_blasint in_pos,
    const libxsmm_blasint offs_in_pos,
    const libxsmm_datatype dtype) {
  // This "singular" type dictates that the arg is a regular tensor (and not a
  // set of tensors)
  libxsmm_matrix_arg_attributes arg_singular_attr =
      libxsmm_create_matrix_arg_attributes(
          LIBXSMM_MATRIX_ARG_TYPE_SINGULAR,
          LIBXSMM_MATRIX_ARG_SET_TYPE_NONE,
          0,
          0);
  // Arg metadata include equation id and pos in arg array at runtime
  libxsmm_matrix_eqn_arg_metadata arg_metadata =
      libxsmm_create_matrix_eqn_arg_metadata(idx, in_pos);
  libxsmm_meqn_arg_shape arg_shape =
      libxsmm_create_meqn_arg_shape(m, n, ld, dtype);
  return libxsmm_matrix_eqn_push_back_arg_v2(
      arg_metadata, arg_shape, arg_singular_attr);
}

inline libxsmm_matrix_eqn_function meqn_dispatch(
    const libxsmm_blasint m,
    const libxsmm_blasint n,
    const libxsmm_blasint* ldo,
    const libxsmm_datatype out_type,
    const unsigned int idx) {
  libxsmm_meqn_arg_shape arg_shape =
      libxsmm_create_meqn_arg_shape(m, n, *ldo, out_type);
  return libxsmm_dispatch_matrix_eqn_v2(idx, arg_shape);
}

inline int meqn_push_unary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_unary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_UNARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  // OP metadata include equation id and an integer dictating where the op
  // metadata at runtime (if any) are located in the op arg array. -1 dictates
  // there are no op metadata needed
  libxsmm_matrix_eqn_op_metadata op_metadata =
      libxsmm_create_matrix_eqn_op_metadata(idx, -1);
  return libxsmm_matrix_eqn_push_back_unary_op_v2(
      op_metadata, type, dtype, flags);
}
inline int meqn_push_binary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_binary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_BINARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  libxsmm_matrix_eqn_op_metadata op_metadata =
      libxsmm_create_matrix_eqn_op_metadata(idx, -1);
  return libxsmm_matrix_eqn_push_back_binary_op_v2(
      op_metadata, type, dtype, flags);
}
inline int meqn_push_ternary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_ternary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_TERNARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  libxsmm_matrix_eqn_op_metadata op_metadata =
      libxsmm_create_matrix_eqn_op_metadata(idx, -1);
  return libxsmm_matrix_eqn_push_back_ternary_op_v2(
      op_metadata, type, dtype, flags);
}

class BaseTPP {
 public:
  void* get_kernel() {
    auto& kernel_cache = get_kernel_cache();
    void* kernel = NULL;
    if (hash == "")
      hash = hash_str();
    auto search = kernel_cache.find(hash);
    if (search != kernel_cache.end())
      kernel = search->second;
    if (kernel == NULL) {
      kernel = build_kernel();
      if (kernel == NULL) {
        fprintf(stderr, "Unable to get JIT kernel for %s\n", hash.c_str());
        exit(1);
      }
      // printf("TPP: %s @ %p\n", hash.c_str(), kernel);
      kernel_cache[hash] = kernel;
    }
    return kernel;
  }
  // We should make hash_str() public
  std::string get_hash_str() {
    return hash_str();
  }

 protected:
  std::unordered_map<std::string, void*>& get_kernel_cache() {
    static std::unordered_map<std::string, void*> kernel_cache;
    return kernel_cache;
  }
  virtual std::string hash_str() = 0;
  virtual void* build_kernel() = 0;
  std::string hash = "";
  bool initialized = false;
};

class UnaryTPP : public BaseTPP {
 public:
  UnaryTPP() {initialized = false;}
  UnaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_unary_type type)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        dt_in(dt_in),
        dt_out(dt_out),
        dt_compute(dt_compute),
        flags(flags),
        type(type) {
    kernel = (libxsmm_meltwfunction_unary)get_kernel();
    initialized = true;
  }

  void operator()(void* in, void* out) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.out.primary = out;
    kernel(&unary_param);
  }
  void operator()(void* in, void* out, void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }
  void operator()(void* in, void* in2, void* in3, void* out, void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = in2;
    unary_param.in.tertiary = in3;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }

  void operator()(
      void* in,
      void* in2,
      void* in3,
      void* op,
      void* op2,
      void* op3,
      void* out,
      void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = in2;
    unary_param.in.tertiary = in3;
    unary_param.op.primary = op;
    unary_param.op.secondary = op2;
    unary_param.op.tertiary = op3;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }

  const libxsmm_blasint get_ldo() const { return ldo;}
  const libxsmm_meltw_unary_type get_type() const { return type;}
  const libxsmm_bitfield get_flags() const { return flags;}

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(
        hash,
        200,
        "unary_r%d_c%d_i%d_o%d_di%d_do%d_dc%d_f%d_t%d",
        rows,
        cols,
        ldi,
        ldo,
        dt_in,
        dt_out,
        dt_compute,
        flags,
        type);
    return std::string(hash);
  }
  void* build_kernel() override {

    libxsmm_meltw_unary_shape shape = libxsmm_create_meltw_unary_shape(
        cols, rows, ldi, ldo, dt_in, dt_out, dt_compute);

    //printf("unary_shape %d %d %d %d \n", shape.m, shape.n, shape.ldi, shape.ldo);
    return (void*)libxsmm_dispatch_meltw_unary_v2(type, shape, flags);
  }

  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  libxsmm_datatype dt_in;
  libxsmm_datatype dt_out;
  libxsmm_datatype dt_compute;
  libxsmm_bitfield flags;
  libxsmm_meltw_unary_type type;
  libxsmm_meltwfunction_unary kernel = NULL;
};

class BinaryTPP : public BaseTPP {
 public:
  BinaryTPP() {}
  BinaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_binary_type type)
      : BinaryTPP(
            rows,
            cols,
            ldi,
            ldi,
            ldo,
            dt_in,
            dt_in,
            dt_out,
            dt_compute,
            flags,
            type) {}
  BinaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi0,
      libxsmm_blasint ldi1,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in0,
      libxsmm_datatype dt_in1,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_binary_type type)
      : rows(rows),
        cols(cols),
        ldi0(ldi0),
        ldi1(ldi1),
        ldo(ldo),
        dt_in0(dt_in0),
        dt_in1(dt_in1),
        dt_out(dt_out),
        dt_compute(dt_compute),
        flags(flags),
        type(type) {
    kernel = (libxsmm_meltwfunction_binary)get_kernel();
    initialized = true;
  }

  void operator()(void* in0, void* in1, void* out) {
    if (!initialized)
      return;
    libxsmm_meltw_binary_param binary_param;
    binary_param.in0.primary = in0;
    binary_param.in1.primary = in1;
    binary_param.out.primary = out;
    kernel(&binary_param);
  }

  const libxsmm_blasint get_ldo() const { return ldo;}
  const libxsmm_datatype get_dt_in0() const { return dt_in0;}
  const libxsmm_datatype get_dt_in1() const { return dt_in1;}
  const libxsmm_meltw_binary_type get_type() const { return type;}
  const libxsmm_bitfield get_flags() const { return flags;}

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(
        hash,
        200,
        "binary_r%d_c%d_i0%d_i1%d_o%d_di0%d_di1%d_do%d_dc%d_f%d_t%d",
        rows,
        cols,
        ldi0,
        ldi1,
        ldo,
        dt_in0,
        dt_in1,
        dt_out,
        dt_compute,
        flags,
        type);
    return std::string(hash);
  }
  void* build_kernel() override {
    libxsmm_meltw_binary_shape shape = libxsmm_create_meltw_binary_shape(
        cols, rows, ldi0, ldi1, ldo, dt_in0, dt_in1, dt_out, dt_compute);
    return (void*)libxsmm_dispatch_meltw_binary_v2(type, shape, flags);
  }

  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi0;
  libxsmm_blasint ldi1;
  libxsmm_blasint ldo;
  libxsmm_datatype dt_in0;
  libxsmm_datatype dt_in1;
  libxsmm_datatype dt_out;
  libxsmm_datatype dt_compute;
  libxsmm_bitfield flags;
  libxsmm_meltw_binary_type type;
  libxsmm_meltwfunction_binary kernel = NULL;
};

template <typename T>
class SetZeroTPP {
 public:
  SetZeroTPP() {}
  SetZeroTPP(int N) : SetZeroTPP(1, N) {}
  SetZeroTPP(int rows, int cols) : SetZeroTPP(rows, cols, cols) {}
  SetZeroTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldo,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_XOR) {}
  void operator()(T* buf) {
    kernel((void*)buf, (void*)buf);
  }
  void ref(T* buf) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        buf[i * ldo + j] = 0;
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout>
class ConvertTPP {
 public:
  ConvertTPP() {}
  ConvertTPP(int N) : ConvertTPP(1, N) {}
  ConvertTPP(int rows, int cols) : ConvertTPP(rows, cols, cols, cols) {}
  ConvertTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY),
        init_done(true) {}
  void operator()(Tin* in, Tout* out) {
    if (!(XsmmDtype<Tin>() == LIBXSMM_DATATYPE_F32 &&
          XsmmDtype<Tout>() == LIBXSMM_DATATYPE_F32) ||
          ((void*)in != (void*)out))
      kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[i * ldi + j];
      }
    }
  }
  bool initialized() {
    return init_done;
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
  bool init_done = false;
};

class ConvertSplitTPP {
 public:
  ConvertSplitTPP() {}
  ConvertSplitTPP(int N)
      : N(N),
        init_done(true) {}
  void operator()(bfloat16* hi, bfloat16* lo, float* out) {
    printf("Error: non-reference implementation for ConvertSplitTPP has not been implemented\n");
    exit(-1);
  }

  void ref(bfloat16* in_hi, bfloat16* in_lo, float* out) {
    auto in_hi_cast = (libxsmm_bfloat16*)in_hi;
    auto in_lo_cast = (libxsmm_bfloat16*)in_lo;
    for (int i = 0; i < N; i++) {
      union libxsmm_bfloat16_f32 bf16_hp;
      bf16_hp.i[0] = in_lo_cast[i];
      bf16_hp.i[1] = in_hi_cast[i];
      //std::cout << "i = " << i << " in_hi[i] = " << in_hi[i] << " in_lo[i] = " << in_lo[i] << std::endl;
      //union libxsmm_bfloat16_f32 bf16_dbg;
      //bf16_dbg.i[0] = 0;
      //bf16_dbg.i[1] = in_hi_cast[i];
      //printf("in_lo_cast[%d] = %hu, in_hi_cast[%d] = %hu combined = %6.6f dbg = %6.6f\n", i, in_lo_cast[i], i, in_hi_cast[i], bf16_hp.f, bf16_dbg.f);
      out[i] = bf16_hp.f;
    }
  }

  bool initialized() {
    return init_done;
  }

 private:
  int N = 0;
  UnaryTPP kernel;
  bool init_done = false;
};


template <typename T>
class CpyTPP {
 public:
  CpyTPP() {}
  CpyTPP(int N) : CpyTPP(1, N) {}
  CpyTPP(int rows, int cols) : CpyTPP(rows, cols, cols, cols) {}
  CpyTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = in[i * ldi + j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class CpyBiasTPP {
 public:
  CpyBiasTPP() {}
  CpyBiasTPP(int rows, int cols) : CpyBiasTPP(rows, cols, cols) {}
  CpyBiasTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            cols,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class CpyBcastTPP {
 public:
  CpyBcastTPP() {}
  CpyBcastTPP(int rows, int cols) : CpyBcastTPP(rows, cols, cols) {}
  CpyBcastTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[i];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};
template <typename T>
class AddBiasTPP {
 public:
  AddBiasTPP() {}
  AddBiasTPP(int rows, int cols) : AddBiasTPP(rows, cols, cols) {}
  AddBiasTPP(int rows, int cols, int ld)
      : rows(rows),
        cols(cols),
        ld(ld),
        kernel(
            rows,
            cols,
            ld,
            ld,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_ADD),
        cvt() {
    if (!std::is_same<T, float>::value)
      cvt = ConvertTPP<T, float>(1, cols);
  }
  void operator()(T* in, float* out) {
    if (std::is_same<T, float>::value) {
      kernel((void*)in, (void*)out, (void*)out);
    } else {
      float tmp[cols];
      cvt(in, tmp);
      kernel((void*)tmp, (void*)out, (void*)out);
    }
  }
  void ref(T* in, float* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ld + c] += (float)in[c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ld;
  BinaryTPP kernel;
  ConvertTPP<T, float> cvt;
};

template <typename T>
class AddBiasConvTPP : public BinaryTPP {
  using BinaryTPP::rows;
  using BinaryTPP::cols;
  using BinaryTPP::ldo;
 public:
  AddBiasConvTPP() : BinaryTPP() {}
  AddBiasConvTPP(int rows, int cols) : AddBiasConvTPP(rows, cols, cols) {}
  AddBiasConvTPP(int rows, int cols, int ld)
      : BinaryTPP(
            rows,
            cols,
            ld,
            ld,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_ADD) {}
  void operator()(T* in, T* out) {
    BinaryTPP::operator()((void*)in, (void*)out, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        float ftmp = out[r * ldo + c];
        ftmp += (float)in[c];
        out[r * ldo + c] = ftmp;
      }
    }
  }
};


template <typename Tin, typename Tout = Tin>
class AddTPP {
 public:
  AddTPP() {}
  AddTPP(int N) : AddTPP(1, N) {}
  AddTPP(int rows, int cols) : AddTPP(rows, cols, cols, cols) {}
  AddTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_ADD) {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi + c] + (float)in1[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin>
class GradBiasTPP {
 public:
  GradBiasTPP() {}
  GradBiasTPP(int rows, int cols) : GradBiasTPP(rows, cols, cols) {}
  GradBiasTPP(int rows, int cols, int ldi)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        reduce(
            rows,
            cols,
            ldi,
            cols,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        add(cols) {}
  void operator()(Tin* in, float* out) {
    float tmp[cols];
    reduce((void*)in, (void*)tmp);
    add(tmp, out, out);
  }
  void ref(Tin* in, float* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c] += (float)in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;

  UnaryTPP reduce;
  AddTPP<float, float> add;
};

template <typename Tin, typename Tout = Tin>
class ReduceAddColTPP {
 public:
  ReduceAddColTPP() {}
  ReduceAddColTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        reduce(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) {}
  void operator()(Tin* in, float* out) {
    reduce(in, out);
  }
  void ref(Tin* in, float* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        if (r == 0)
          out[c] = 0;
        out[c] += (float)in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi, ldo;

  UnaryTPP reduce;
};

template <typename Tin, typename Tout = Tin>
class ReduceAddRowTPP {
 public:
  ReduceAddRowTPP() {}
  ReduceAddRowTPP(int rows, int cols, bool acc)
      : ReduceAddRowTPP(rows, cols, cols, acc) {}
  ReduceAddRowTPP(int rows, int cols, int ldi, bool acc)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        acc(acc),
        reduce(
            rows,
            cols,
            ldi,
            cols,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        add(rows) {}
  void operator()(Tin* in, Tout* out) {
    if (acc) {
      Tout tmp[rows];
      reduce((void*)in, (void*)tmp);
      add(tmp, out, out);
    } else {
      reduce((void*)in, (void*)out);
    }
  }
  void ref(Tin* in, Tout* out) {
    for (int r = 0; r < rows; r++) {
      if (!acc) {
        out[r] = 0;
      }
      for (int c = 0; c < cols; c++) {
        out[r] += (float)in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  bool acc;
  UnaryTPP reduce;
  AddTPP<Tout, Tout> add;
};

template <typename T>
class BCastMulTPP {
 public:
  BCastMulTPP() {}
  BCastMulTPP(int rows, int cols) : BCastMulTPP(rows, cols, cols) {}
  BCastMulTPP(int rows, int cols, int ld)
      : rows(rows),
        cols(cols),
        ld(ld),
        kernel(
            rows,
            cols,
            ld,
            ld,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL),
        cvt() {
    if (!std::is_same<T, T>::value)
      cvt = ConvertTPP<T, T>(1, rows);
  }
  void operator()(T* in, T* out) {
    if (std::is_same<T, T>::value) {
      kernel((void*)in, (void*)out, (void*)out);
    } else {
      T tmp[rows];
      cvt(in, tmp);
      kernel((void*)tmp, (void*)out, (void*)out);
    }
  }
  void ref(T* in, T* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c * ld + r] *= (T)in[r];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ld;
  BinaryTPP kernel;
  ConvertTPP<T, T> cvt;
};

template <typename Tin, typename Tout>
class ScaleTPP {
 public:
  ScaleTPP() {}
  ScaleTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in, Tout* out, float scale) {
    Tin alpha = scale;
    kernel((void*)&alpha, (void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out, float scale) {
    Tin alpha = scale;
    for (int i = 0; i < N; i++) {
      out[i] = (float)in[i] * (float)alpha;
    }
  }

 private:
  int N = 0;
  BinaryTPP kernel;
};

template <typename T, typename TN = float>
class Norm2TPP {
 public:
  Norm2TPP() {}
  Norm2TPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) {}
  void operator()(T* in, TN* sum) {
    float lsum = 0.0f;
    kernel((void*)in, (void*)&lsum);
    *sum += (TN)lsum;
  }
  void ref(T* in, TN* sum) {
    float lsum = 0.0f;
    for (int i = 0; i < N; i++) {
      lsum += (float)in[i] * (float)in[i];
    }
    *sum += (TN)lsum;
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename T>
class RecpTPP {
 public:
  RecpTPP() {}
  RecpTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < N; i++)
      out[i] = 1.0 / in[i];
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename T>
class RecpSqrtTPP {
 public:
  RecpSqrtTPP() {}
  RecpSqrtTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < N; i++)
      out[i] = 1.0 / sqrt(in[i]);
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class MulNormTPP {
 public:
  MulNormTPP() {}
  MulNormTPP(int rows, int cols) : MulNormTPP(rows, cols, cols, cols) {}
  MulNormTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1, // ldi0
            ldi, // ldi1
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in, Tin* in2, Tout* out) {
    kernel((void*)in, (void*)in2, (void*)out);
  }
  void ref(Tin* in, Tin* in2, Tout* out) {
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
        out[r * ldo + c] = in[r] * in2[r * ldi + c];
  }

 private:
  int rows, cols;
  int ldi, ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout>
class ScaleAddTPP {
 public:
  ScaleAddTPP() {}
  ScaleAddTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MULADD) {}
  void operator()(Tin* in, Tout* out, float scale) {
    Tin alpha = scale;
    kernel((void*)&alpha, (void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out, float scale) {
    Tin alpha = scale;
    for (int i = 0; i < N; i++) {
      out[i] += (float)in[i] * (float)alpha;
    }
  }

 private:
  int N = 0;
  BinaryTPP kernel;
};

template <typename Tin, typename Tind, typename Tout>
class EmbeddingFwdTPP {
 public:
  EmbeddingFwdTPP() {}
  EmbeddingFwdTPP(int rows, int cols, int ldi)
      : EmbeddingFwdTPP(rows, cols, ldi, ldi) {}
  EmbeddingFwdTPP(int rows, int cols)
      : EmbeddingFwdTPP(rows, cols, cols, cols) {}
  EmbeddingFwdTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            (LIBXSMM_MELTW_FLAG_UNARY_GS_COLS |
             (sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES
                                : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES)),
            LIBXSMM_MELTW_TYPE_UNARY_GATHER) {}
  void operator()(Tin* in0, Tind* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, NULL, (void*)out, NULL);
  }
  void ref(Tin* in0, Tind* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      auto ind = in1[r];
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = in0[ind * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi = 0;
  int ldo = 0;
  UnaryTPP kernel;
};

template <typename Tin, typename Tind, typename Tout>
class EmbeddingBwdTPP {
 public:
  EmbeddingBwdTPP() {}
  EmbeddingBwdTPP(int E)
      : E(E),
        kernel(
            0,
            E,
            E,
            E,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            (libxsmm_meltw_unary_flags)(
                sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES
                                  : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES),
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD) {}
  void operator()(Tin* in0, Tind* in1, Tout* out, int N) {
    unsigned long long _N = N;
    kernel((void*)in0, (void*)in1, (void*)&_N, (void*)out, NULL);
  }
  void ref(Tin* in0, Tind* in1, Tout* out, int N) {
    for (long v = 0; v < E; v++)
      out[v] = 0;
    for (long s = 0; s < N; s++) {
      auto ind = in1[s];
      for (long v = 0; v < E; v++)
        out[v] += in0[ind * E + v];
    }
  }

 private:
  int E = 0;
  UnaryTPP kernel;
};

class XformTPP {
 public:
  XformTPP() {}
  XformTPP(
      libxsmm_blasint rows_i,
      libxsmm_blasint cols_i,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dtype,
      libxsmm_meltw_unary_type type)
      : rows(rows_i),
        cols(cols_i),
        ldi(ldi),
        ldo(ldo),
        dtype(dtype),
        type(type),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            dtype,
            dtype,
            dtype,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            type) {}
  void operator()(void* in, void* out) {
    kernel(in, out);
  }
  typedef enum XFORM_TYPE {
    XFORM_NONE_TPP = 0,
    XFORM_XPOSE_TPP = 1,
    XFORM_N2V_TPP = 2,
    XFORM_XPOSE_N2V_TPP = 3,
    XFORM_XPOSE_V2V_TPP = 4,
    XFORM_N2V_PAD_TPP   = 5
  } XFORM_TYPE;

 private:
  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  libxsmm_datatype dtype;
  libxsmm_meltw_unary_type type;
  UnaryTPP kernel;
};

template <typename T>
class XformExtTPP {
 public:
  XformExtTPP() {}
  XformExtTPP(
      /* rows and cols as for input tensor */
      int rows,
      int cols,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
      : XformExtTPP(
            rows,
            cols,
            (xtype == XformTPP::XFORM_N2V_TPP ? rows : cols),
            (xtype == XformTPP::XFORM_N2V_TPP ? cols : rows),
            xtype,
            ignore_vnni_for_fp32) {}
  XformExtTPP(
      int in_rows,
      int in_cols,
      int out_rows,
      int out_cols,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
      : XformExtTPP(
            in_rows,
            in_cols,
            out_rows,
            out_cols,
            in_cols,
            out_cols,
            xtype,
            ignore_vnni_for_fp32) {}
  XformExtTPP(
      int in_rows,
      int in_cols,
      int out_rows,
      int out_cols,
      int ldi,
      int ldo,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
      : in_rows(in_rows),
        in_cols(in_cols),
        out_rows(out_rows),
        out_cols(out_cols),
        ldi(ldi),
        ldo(ldo),
        xtype(xtype),
        dtype(XsmmDtype<T>()),
        kernel(),
        cvt(),
        cpy(),
        zero() {
    libxsmm_meltw_unary_type unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
    if (ignore_vnni_for_fp32 == false) {
      TPP_ASSERT(
          (xtype == XformTPP::XFORM_XPOSE_TPP || dtype != LIBXSMM_DATATYPE_F32),
          "Only Transpose Xofrm supportd for FP32 datatype, specified %d\n",
          (int)xtype);
    }
    const int BS = xsmm_get_vnni_block_size(dtype);
    if (xtype == XformTPP::XFORM_N2V_TPP || xtype == XformTPP::XFORM_N2V_PAD_TPP) {
      in_rows_p = out_rows;
      in_cols_p = out_cols;

      if (dtype != LIBXSMM_DATATYPE_F32) {
        if (xtype == XformTPP::XFORM_N2V_TPP) {
          if (BS == 1) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
          } else if (BS == 2) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2;
          } else if (BS == 4) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4;
          } else {
            TPP_ASSERT(false, "N2VTPP: unsupported packing size (%d)\n", BS);
          }
          if (in_rows_p % BS != 0) {
            printf("Got you!\n");
          }
          TPP_ASSERT(in_rows_p % BS == 0, "N2VTPP: unaligned number of rows\n");
        }
        else { /* XformTPP::XFORM_N2V_PAD_TPP */
          if (BS == 2) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD;
          } else if (BS == 4) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD;
          } else {
            TPP_ASSERT(false, "N2VTPP: unsupported packing size (%d)\n", BS);
          }
          TPP_ASSERT(in_rows_p % BS != 0, "N2VTPP_PAD: aligned number of rows, N2V_PAD should not be used\n");
        }
      } else {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
      }

    } else {
      in_rows_p = out_cols;
      in_cols_p = out_rows;
      if (dtype != LIBXSMM_DATATYPE_F32) {
        if (xtype == XformTPP::XFORM_XPOSE_TPP) {
          unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
        } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
          // unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT;
          unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
          TPP_ASSERT(
              in_cols_p % BS == 0, "XposeN2VTPP: uneven number of cols\n");
        } else {
          if (BS == 2) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T;
          } else if (BS == 4) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T;
          } else {
            TPP_ASSERT(false, "V2VTPP: unsupported packing size (%d)\n", BS);
          }
          TPP_ASSERT(in_rows % BS == 0, "XposeV2VTPP: uneven number of rows\n");
          TPP_ASSERT(
              in_cols_p % BS == 0, "XposeV2VTPP: uneven number of cols\n");
        }
      } else {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
      }
    }
    TPP_ASSERT(
        (in_rows_p >= in_rows && in_cols_p >= in_cols),
        "Invalid output rows or cols value\n");
    TPP_ASSERT(
        in_rows_p == in_rows || in_cols_p == in_cols,
        "Padding can only be done in rows or cols\n");

    if (xtype != XformTPP::XFORM_XPOSE_N2V_TPP) {
      int ld = (in_rows_p != in_rows || in_cols_p != in_cols) ? in_cols_p : ldi;
      kernel = XformTPP(in_rows_p, in_cols_p, ld, ldo, dtype, unary_type);
    } else {
      // LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT not implemented so use
      // workaround...
      kernel = XformTPP(
          in_rows_p,
          in_cols_p / BS,
          ldi / BS,
          ldo,
          ((dtype == LIBXSMM_DATATYPE_BF16 && BS == 4) ||
           (dtype == LIBXSMM_DATATYPE_BF8 && BS == 8))
              ? LIBXSMM_DATATYPE_F64
              : LIBXSMM_DATATYPE_F32,
          unary_type);
    }

    if ((xtype == XformTPP::XFORM_N2V_TPP ||
         xtype == XformTPP::XFORM_XPOSE_TPP) &&
        in_rows_p != in_rows) {
      cpy = CpyTPP<T>(in_rows, in_cols, ldi, in_cols);
      zero = SetZeroTPP<T>(in_rows_p - in_rows, in_cols);
      zero_offset = in_rows * in_cols;
    } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP && in_cols_p != in_cols) {
      cpy = CpyTPP<T>(in_rows, in_cols, ldi, in_cols_p);
      zero = SetZeroTPP<T>(in_rows, in_cols_p - in_cols, in_cols_p);
      zero_offset = in_cols;
    } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP && in_cols_p != in_cols) {
      cpy = CpyTPP<T>(in_rows / BS, in_cols * BS, ldi * BS, in_cols_p * BS);
      zero = SetZeroTPP<T>(
          in_rows / BS, (in_cols_p - in_cols) * BS, in_cols_p * BS);
      zero_offset = in_cols * BS;
    }
    if (std::is_same<T, bfloat16>::value)
      cvt = ConvertTPP<float, bfloat16>(in_rows, in_cols);
  }
  void operator()(T* in, T* out) {
    if (in != out) {
      if (in_rows_p != in_rows || in_cols_p != in_cols) {
        T tmp[in_rows_p * in_cols_p];
        cpy(in, tmp);
        zero(tmp + zero_offset);
        kernel((void*)tmp, (void*)out);
      } else {
        kernel((void*)in, (void*)out);
      }
    }
  }
  void ref(T* in, T* out) {
    const int BS = xsmm_get_vnni_block_size(dtype);
    if (xtype == XformTPP::XFORM_XPOSE_TPP) {
      for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
          out[i * ldo + j] = in[j * ldi + i];
        }
      }
    } else if (xtype == XformTPP::XFORM_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_rows) {
              out[i * ldo * BS + j * BS + k] = in[i * ldi * BS + k * ldi + j];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_cols) {
              out[i * ldo * BS + j * BS + k] = in[j * ldi + i * BS + k];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP) {
      for (int j = 0; j < out_rows / BS; j++) {
        for (int i = 0; i < in_rows / BS; i++) {
          for (int k = 0; k < BS; k++) { // RBS
            for (int l = 0; l < BS; l++) { // CBS
              if (j * BS + l < in_cols && i * BS + k < out_cols) {
                out[j * ldo * BS + i * BS * BS + k * BS + l] =
                    in[i * ldi * BS + j * BS * BS + l * BS + k];
              } else {
                out[j * ldo * BS + i * BS * BS + k * BS + l] = 0;
              }
            }
          }
        }
      }
    } else {
      TPP_ASSERT(false, "Should not come here\n");
    }
  }

  void operator()(float* in, bfloat16* out) {
    bfloat16 tmp2[in_rows * in_cols];
    cvt(in, tmp2);
    if (in_rows_p != in_rows || in_cols_p != in_cols) {
      T tmp[in_rows_p * in_cols_p];
      cpy(tmp2, tmp);
      zero(tmp + zero_offset);
      kernel((void*)tmp, (void*)out);
    } else {
      kernel((void*)tmp2, (void*)out);
    }
  }
  void ref(float* in, bfloat16* out) {
    auto BS = xsmm_get_vnni_block_size(dtype);
    if (xtype == XformTPP::XFORM_XPOSE_TPP) {
      for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
          out[i * ldo + j] = in[j * ldi + i];
        }
      }
    } else if (xtype == XformTPP::XFORM_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_rows) {
              out[i * ldo * BS + j * BS + k] = in[i * ldi * BS + k * ldi + j];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_cols) {
              out[i * ldo * BS + j * BS + k] = in[j * ldi + i * BS + k];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP) {
      for (int j = 0; j < out_rows / BS; j++) {
        for (int i = 0; i < out_cols / BS; i++) {
          for (int k = 0; k < BS; k++) { // RBS
            for (int l = 0; l < BS; l++) { // CBS
              if (j * BS + l < in_cols) {
                out[j * ldo * BS + i * BS * BS + k * BS + l] =
                    in[i * ldi * BS + j * BS * BS + l * BS + k];
              } else {
                out[j * ldo * BS + i * BS * BS + k * BS + l] = 0;
              }
            }
          }
        }
      }
    } else {
      TPP_ASSERT(false, "Should not come here\n");
    }
  }
  void operator()(int count, long str_in, long str_out, T* in, T* out) {
    for (int i = 0; i < count; i++) {
      this->operator()(&in[i * str_in], &out[i * str_out]);
    }
  }
  void ref(int count, long str_in, long str_out, T* in, T* out) {
    for (int i = 0; i < count; i++) {
      this->ref(&in[i * str_in], &out[i * str_out]);
    }
  }
  void operator()(
      int count,
      long str_in,
      long str_out,
      float* in,
      bfloat16* out) {
    for (int i = 0; i < count; i++) {
      this->operator()(&in[i * str_in], &out[i * str_out]);
    }
  }
  void ref(int count, long str_in, long str_out, float* in, bfloat16* out) {
    for (int i = 0; i < count; i++) {
      this->ref(&in[i * str_in], &out[i * str_out]);
    }
  }

 private:
  libxsmm_blasint in_rows = 0;
  libxsmm_blasint in_cols = 0;
  libxsmm_blasint out_rows = 0;
  libxsmm_blasint out_cols = 0;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  int in_rows_p = 0;
  int in_cols_p = 0;
  XformTPP::XFORM_TYPE xtype;
  libxsmm_datatype dtype;
  int zero_offset = 0;
  XformTPP kernel;
  ConvertTPP<float, bfloat16> cvt;
  CpyTPP<T> cpy;
  SetZeroTPP<T> zero;
};

template <typename Tin, typename Tout>
class BrgemmBaseTPP {
 public:
  BrgemmBaseTPP() {}
  BrgemmBaseTPP(
      long M,
      long N,
      long K,
      bool reduce_offset,
      long str_a,
      long str_b,
      long lda,
      long ldb,
      long ldc,
      float beta,
      int a_trans,
      int c_vnni,
      int unroll_hint)
      : M(M),
        N(N),
        K(K),
        reduce_offset(reduce_offset),
        str_a(str_a),
        str_b(str_b),
        lda(lda),
        ldb(ldb),
        ldc(ldc),
        beta(beta),
        a_trans(a_trans),
        c_vnni(c_vnni),
        unroll_hint(unroll_hint),
        is_gemm_ext(false),
        k_gemm_with_tc(this, 0),
        k_cfg(this, 1),
        k_rls(this, 2),
        k_gemm_no_tc(this, 3) {}

  BrgemmBaseTPP(
      long M,
      long N,
      long K,
      bool reduce_offset,
      long str_a,
      long str_b,
      long lda,
      long ldb,
      long ldc,
      float beta,
      int a_trans,
      int c_vnni,
      int unroll_hint,
      UnaryTPP  *argop_a,
      UnaryTPP  *argop_b,
      UnaryTPP  *argop_c,
      BinaryTPP *postop)
      : M(M),
        N(N),
        K(K),
        reduce_offset(reduce_offset),
        str_a(str_a),
        str_b(str_b),
        lda(lda),
        ldb(ldb),
        ldc(ldc),
        beta(beta),
        a_trans(a_trans),
        c_vnni(c_vnni),
        unroll_hint(unroll_hint),
        is_gemm_ext(true),
        has_argop_a(argop_a == NULL ? false : true),
        has_argop_b(argop_b == NULL ? false : true),
        has_argop_c(argop_c == NULL ? false : true),
        has_postop(postop == NULL ? false : true),
        argop_a(argop_a),
        argop_b(argop_b),
        argop_c(argop_c),
        postop(postop),
        k_ext_gemm_with_tc(this, 0),
        k_ext_cfg(this, 1),
        k_ext_rls(this, 2),
        k_ext_gemm_no_tc(this, 3) {}

  void config() {
    if (!is_gemm_ext)
      k_cfg(NULL);
    else
      k_ext_cfg(NULL);
  }
  void release() {
    if (!is_gemm_ext)
      k_rls(NULL);
    else
      k_ext_rls(NULL);
    //k_rls(NULL);
  }

  long flops() {
    return 2L * M * N * K;
  }

  class BrgemmKernel : public BaseTPP {
   public:
    BrgemmKernel() {}
    BrgemmKernel(BrgemmBaseTPP* p, int config) : p(p), config(config) {
      auto dt_in = XsmmDtype<Tin>();
      auto dt_out = XsmmDtype<Tout>();
      long type = -1;
      if (dt_in == LIBXSMM_DATATYPE_F32) {
        TPP_ASSERT(dt_out == LIBXSMM_DATATYPE_F32, "BRGEMM Assert\n");
        type = 0;
      } else if (dt_out == LIBXSMM_DATATYPE_F32) {
        type = 1;
      } else {
        type = 2;
      }
      if (type != 0)
        TPP_ASSERT(
            p->a_trans == 0, "A Transpose supported only for FP32 BRGEMM\n");
      brgemm_type = type;
      kernel.gemm = (libxsmm_gemmfunction)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_gemm_param* gemm_param) {
      TPP_ASSERT(initialized, "Attempt to call uninitialized BrgemmKernel\n");
      kernel.gemm(gemm_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "brgemm_m%ld_n%ld_k%ld_offset%d_a%ld_b%ld_t%ld_beta%d_at%d_cv%d_uh%d_ld_a%ld_b%ld_c%ld_cfg%d",
          p->M,
          p->N,
          p->K,
          p->reduce_offset,
          p->str_a,
          p->str_b,
          brgemm_type,
          (int)p->beta,
          p->a_trans,
          p->c_vnni,
          p->unroll_hint,
          (long)p->lda,
          (long)p->ldb,
          (long)p->ldc,
          config);
      return std::string(hash);
    }
    void* build_kernel() override {
      // float alpha = 1.0;
      libxsmm_gemm_shape l_shape;
      libxsmm_gemm_batch_reduce_config l_brconfig;
      libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
      libxsmm_bitfield l_prefetch_flags = 0;
      libxsmm_xmmfunction l_test_jit = {NULL};

      if (p->a_trans == 1)
        l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
      if (brgemm_type != 0)
        l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
      if (p->beta == 0)
        l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
      if (p->c_vnni == 1)
        l_flags |= LIBXSMM_GEMM_FLAG_VNNI_C;

      // config = 0 - normal
      // config = 1 - no tile release
      // config = 2 - no tile config
      // config = 3 - brgemm with no tile config or release
      if (config == 1) {
        l_flags |= LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG;
      } else if (config == 2) {
        l_flags |= LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
      } else if (config == 3) {
        l_flags |=
            (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG |
             LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
      }

      /* setting update GEMM struct */
      l_shape.m = p->N;
      l_shape.n = p->M;
      l_shape.k = p->K;
      l_shape.lda = p->ldb;
      l_shape.ldb = p->lda;
      l_shape.ldc = p->ldc;
      l_shape.a_in_type = XsmmDtype<Tin>();
      l_shape.b_in_type = XsmmDtype<Tin>();
      l_shape.out_type = XsmmDtype<Tout>();
      l_shape.comp_type = LIBXSMM_DATATYPE_F32;

      if (p->reduce_offset) {
        l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
        l_brconfig.br_stride_a_hint = 0;
        l_brconfig.br_stride_b_hint = 0;
      } else {
        l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
        l_brconfig.br_stride_a_hint = p->str_b * sizeof(Tin);
        l_brconfig.br_stride_b_hint = p->str_a * sizeof(Tin);
      }
      l_brconfig.br_unroll_hint = p->unroll_hint;

      l_test_jit.gemm = libxsmm_dispatch_brgemm_v2(
          l_shape, l_flags, l_prefetch_flags, l_brconfig);

      return (void*)l_test_jit.gemm;
    }

   private:
    BrgemmBaseTPP* p;
    int config;
    libxsmm_xmmfunction kernel;
    long brgemm_type = -1;
  };

  class BrgemmExtKernel : public BaseTPP {
   public:
    BrgemmExtKernel() {}
    BrgemmExtKernel(BrgemmBaseTPP* p, int config) : p(p), config(config) {
      auto dt_in = XsmmDtype<Tin>();
      auto dt_out = XsmmDtype<Tout>();
      long type = -1;
      if (dt_in == LIBXSMM_DATATYPE_F32) {
        TPP_ASSERT(dt_out == LIBXSMM_DATATYPE_F32, "BRGEMM Assert\n");
        type = 0;
      } else if (dt_out == LIBXSMM_DATATYPE_F32) {
        type = 1;
      } else {
        type = 2;
      }
      if (type != 0)
        TPP_ASSERT(
            p->a_trans == 0, "A Transpose supported only for FP32 BRGEMM\n");
      brgemm_type = type;
      kernel.gemm_ext = (libxsmm_gemmfunction_ext)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_gemm_ext_param* gemm_ext_param) {
      TPP_ASSERT(initialized, "Attempt to call uninitialized BrgemmExtKernel\n");
      kernel.gemm_ext(gemm_ext_param);
    }

   protected:
    std::string hash_str() override {
      char hash[400];
      snprintf(
          hash,
          400,
          "brgemmext_m%ld_n%ld_k%ld_offset%d_a%ld_b%ld_t%ld_beta%d_at%d_cv%d_uh%d_ld_a%ld_b%ld_c%ld_cfg%d_ap_ld%d_t%d_f%d_s%d_bp_ld%d_t%d_f%d_s%d_cp_ld%d_t%d_f%d_s%d_po_ld%d_dt%d_t%d_f%d",
          p->M,
          p->N,
          p->K,
          p->reduce_offset,
          p->str_a,
          p->str_b,
          brgemm_type,
          (int)p->beta,
          p->a_trans,
          p->c_vnni,
          p->unroll_hint,
          (long)p->lda,
          (long)p->ldb,
          (long)p->ldc,
          config,
          p->argop_a->get_ldo(),
          p->argop_a->get_type(),
          p->argop_a->get_flags(),
          0 /* argop_a: store_ap */,
          p->argop_b->get_ldo(),
          p->argop_b->get_type(),
          p->argop_b->get_flags(),
          0 /* argop_b: store_bp */,
          p->argop_c->get_ldo(),
          p->argop_c->get_type(),
          p->argop_c->get_flags(),
          0 /* argop_c: store_cp */,
          p->postop->get_ldo(),
          p->postop->get_dt_in1(), /* assumes that dt_in1 == dt_in0 */
          p->postop->get_type(),
          p->postop->get_flags());
      return std::string(hash);
    }
    void* build_kernel() override {
      // float alpha = 1.0;
      libxsmm_gemm_shape l_shape;
      libxsmm_gemm_batch_reduce_config l_brconfig;
      libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
      libxsmm_bitfield l_prefetch_flags = 0;
      libxsmm_xmmfunction l_test_jit = {NULL};

      if (p->a_trans == 1)
        l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
      if (brgemm_type != 0)
        l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
      if (p->beta == 0)
        l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
      if (p->c_vnni == 1)
        l_flags |= LIBXSMM_GEMM_FLAG_VNNI_C;

      // config = 0 - normal
      // config = 1 - no tile release
      // config = 2 - no tile config
      // config = 3 - brgemm with no tile config or release
      if (config == 1) {
        l_flags |= LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG;
      } else if (config == 2) {
        l_flags |= LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
      } else if (config == 3) {
        l_flags |=
            (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG |
             LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
      }

      /* setting update GEMM struct */
      l_shape.m = p->N;
      l_shape.n = p->M;
      l_shape.k = p->K;
      l_shape.lda = p->ldb;
      l_shape.ldb = p->lda;
      l_shape.ldc = p->ldc;
      l_shape.a_in_type = XsmmDtype<Tin>();
      l_shape.b_in_type = XsmmDtype<Tin>();
      l_shape.out_type = XsmmDtype<Tout>();
      l_shape.comp_type = LIBXSMM_DATATYPE_F32;

      if (p->reduce_offset) {
        l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
        l_brconfig.br_stride_a_hint = 0;
        l_brconfig.br_stride_b_hint = 0;
      } else {
        l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
        l_brconfig.br_stride_a_hint = p->str_b * sizeof(Tin);
        l_brconfig.br_stride_b_hint = p->str_a * sizeof(Tin);
      }
      l_brconfig.br_unroll_hint = p->unroll_hint;

      libxsmm_gemm_ext_unary_argops l_argops;
      memset( &l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops) );

      libxsmm_gemm_ext_binary_postops l_postops;
      memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );

      if (p->has_argop_a) {
        TPP_ASSERT(false, "Error: argop for A is not supported in BrgemmExtKernel\n");
      }
      if (p->has_argop_b) {
        TPP_ASSERT(false, "Error: argop for B is not supported in BrgemmExtKernel\n");
      }
      if (p->has_argop_c) {
        TPP_ASSERT(false, "Error: argop for C is not supported in BrgemmExtKernel\n");
        l_argops.cp_unary_type  = p->argop_c->get_type();//LIBXSMM_MELTW_TYPE_UNARY_RELU;
        l_argops.ldcp           = p->argop_c->get_ldo();//l_shape.ldc;
      }

//200~LIBXSMM_API libxsmm_gemm_ext_unary_argops libxsmm_create_gemm_ext_unary_argops( const libxsmm_blasint ldap, const libxsmm_meltw_unary_type ap_unary_type, const libxsmm_bitfield ap_unary_flags, const libxsmm_bl
//                                                                                const libxsmm_blasint ldbp, const libxsmm_meltw_unary_type bp_unary_type, const libxsmm_bitfield bp_unary_flags, const libxsmm_bl
//                                                                                const libxsmm_blasint ldcp, const libxsmm_meltw_unary_type cp_unary_type, const libxsmm_bitfield cp_unary_flags, const libxsmm_bl
//LIBXSMM_API libxsmm_gemm_ext_binary_postops libxsmm_create_gemm_ext_binary_postops( const libxsmm_blasint ldd, const libxsmm_datatype d_in_type, const libxsmm_meltw_binary_type d_binary_type, const libxsmm_bit201~

      if (p->has_postop) {
        TPP_ASSERT(p->postop->get_dt_in0() == p->postop->get_dt_in1(), "Error: postop dtype for input arguments differ which is not supported in BrgemmExtKernel\n");
        l_postops = libxsmm_create_gemm_ext_binary_postops(p->postop->get_ldo(), p->postop->get_dt_in1(), p->postop->get_type(), p->postop->get_flags());//LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0);
      }

/*
      if (with_bias)
        l_postops = libxsmm_create_gemm_ext_binary_postops(bk, dtype, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0);
      if (with_relu) {
        l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
        l_argops.ldcp           = l_shape.ldc;
      }
*/

      l_test_jit.gemm_ext = libxsmm_dispatch_brgemm_ext_v2(
          l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops);

      return (void*)l_test_jit.gemm_ext;
    }

   private:
    BrgemmBaseTPP* p;
    int config;
    libxsmm_xmmfunction kernel;
    long brgemm_type = -1;
  };


 protected:
  long M, N, K;
 private:
  bool reduce_offset;
 protected:
  long str_a, str_b;
  libxsmm_blasint lda;
  libxsmm_blasint ldb;
  libxsmm_blasint ldc;
  float beta;
  int a_trans;
  int c_vnni;
 private:
  long brgemm_type = -1;
  int unroll_hint;
 protected:
  BrgemmKernel k_gemm_with_tc;
  BrgemmKernel k_cfg;
  BrgemmKernel k_rls;
  BrgemmKernel k_gemm_no_tc;
 protected: /* FIXME, not sure about the correct specifiers */
  bool   is_gemm_ext = false;
  bool   has_argop_a = false;
  bool   has_argop_b = false;
  bool   has_argop_c = false;
  bool   has_postop  = false;
  UnaryTPP    *argop_a = NULL;
  UnaryTPP    *argop_b = NULL;
  UnaryTPP    *argop_c = NULL;
  BinaryTPP   *postop  = NULL;
  BrgemmExtKernel k_ext_gemm_with_tc;
  BrgemmExtKernel k_ext_cfg;
  BrgemmExtKernel k_ext_rls;
  BrgemmExtKernel k_ext_gemm_no_tc;
};


template <typename Tin, typename Tout>
class BrgemmTPP : public BrgemmBaseTPP<Tin,Tout> {
 using BrgemmBaseTPP<Tin,Tout>::M;
 using BrgemmBaseTPP<Tin,Tout>::N;
 using BrgemmBaseTPP<Tin,Tout>::K;
 using BrgemmBaseTPP<Tin,Tout>::str_a;
 using BrgemmBaseTPP<Tin,Tout>::str_b;
 using BrgemmBaseTPP<Tin,Tout>::lda;
 using BrgemmBaseTPP<Tin,Tout>::ldb;
 using BrgemmBaseTPP<Tin,Tout>::ldc;
 using BrgemmBaseTPP<Tin,Tout>::beta;
 using BrgemmBaseTPP<Tin,Tout>::a_trans;
 using BrgemmBaseTPP<Tin,Tout>::k_gemm_with_tc;
 using BrgemmBaseTPP<Tin,Tout>::k_gemm_no_tc;
 public:
  BrgemmTPP() : BrgemmBaseTPP<Tin,Tout>() {}
  BrgemmTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      float beta = 1.0,
      int a_trans = 0,
      int unroll_hint = 0)
      : BrgemmTPP(
            M,
            N,
            K,
            str_a,
            str_b,
            (a_trans == 0 ? K : M),
            N,
            N,
            beta,
            a_trans,
            unroll_hint) {}
  BrgemmTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      long lda,
      long ldb,
      long ldc,
      float beta,
      int a_trans,
      int unroll_hint)
      : BrgemmTPP(
            M,
            N,
            K,
            str_a,
            str_b,
            (a_trans == 0 ? K : M),
            N,
            N,
            beta,
            a_trans,
            0 /*c_vnni*/,
            unroll_hint) {}
  BrgemmTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      long lda,
      long ldb,
      long ldc,
      float beta,
      int a_trans,
      int c_vnni,
      int unroll_hint)
      : BrgemmBaseTPP<Tin,Tout>(M, N, K, false, str_a, str_b, lda, ldb, ldc, beta, a_trans, c_vnni, unroll_hint) {}

  void operator()(
      Tin* A,
      Tin* B,
      Tout* C,
      unsigned long long count,
      bool no_tile_cfg = false) {
    libxsmm_gemm_param gemm_param;
    memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
    gemm_param.op.tertiary = &count;
    gemm_param.c.primary = (void*)C;
    gemm_param.a.primary = (void*)B;
    gemm_param.b.primary = (void*)A;
    if (!no_tile_cfg) {
      k_gemm_with_tc(&gemm_param);
    } else {
      k_gemm_no_tc(&gemm_param);
    }
  }

  void ref(
      Tin* A,
      Tin* B,
      Tout* C,
      unsigned long long count,
      bool no_tile_cfg = false) {
    auto dtype = XsmmDtype<Tin>();
    for (uint64_t c = 0; c < count; c++) {
      auto A_ = &A[c * str_a];
      auto B_ = &B[c * str_b];
      if (std::is_same<Tin, float>::value) {
        for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++) {
            if (beta == 0.0 && c == 0)
              C[i * N + j] = 0.0;
            for (int k = 0; k < K; k++) {
              if (a_trans == 1) {
                C[i * ldc + j] += A_[k * lda + i] * B_[k * ldb + j];
              } else {
                C[i * ldc + j] += A_[i * lda + k] * B_[k * ldb + j];
              }
            }
          }
        }
      } else {
        const int BS = xsmm_get_vnni_block_size(dtype);
        for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            if (beta == 1.0 && c == 0)
              sum = C[i * ldc + j];
            for (int k = 0; k < K / BS; k++) {
              for (int b = 0; b < BS; b++) {
                sum += (float)A_[i * lda + k * BS + b] *
                    (float)B_[k * ldb * BS + j * BS + b];
              }
            }
            C[i * ldc + j] = (Tout)sum;
          }
        }
      }
    }
  }
};

template <typename Tin, typename Tout>
class BrgemmExtConvTPP : public BrgemmBaseTPP<Tin,Tout> {
 using BrgemmBaseTPP<Tin,Tout>::M;
 using BrgemmBaseTPP<Tin,Tout>::N;
 using BrgemmBaseTPP<Tin,Tout>::K;
 using BrgemmBaseTPP<Tin,Tout>::str_a;
 using BrgemmBaseTPP<Tin,Tout>::str_b;
 using BrgemmBaseTPP<Tin,Tout>::lda;
 using BrgemmBaseTPP<Tin,Tout>::ldb;
 using BrgemmBaseTPP<Tin,Tout>::ldc;
 using BrgemmBaseTPP<Tin,Tout>::beta;
 using BrgemmBaseTPP<Tin,Tout>::a_trans;
 using BrgemmBaseTPP<Tin,Tout>::has_postop;
 using BrgemmBaseTPP<Tin,Tout>::k_ext_gemm_with_tc;
 using BrgemmBaseTPP<Tin,Tout>::k_ext_gemm_no_tc;
 public:
  BrgemmExtConvTPP() : BrgemmBaseTPP<Tin,Tout>() {}
  BrgemmExtConvTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      UnaryTPP  *argop_a,
      UnaryTPP  *argop_b,
      UnaryTPP  *argop_c,
      BinaryTPP *postop)
      : BrgemmExtConvTPP(
            M,
            N,
            K,
            str_a,
            str_b,
            (a_trans == 0 ? K : M),
            N,
            N,
            1.0 /*beta */,
            0 /*a_trans */,
            0 /*unroll_hint */,
            argop_a,
            argop_b,
            argop_c,
            postop) {}
  BrgemmExtConvTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      long lda,
      long ldb,
      long ldc,
      UnaryTPP  *argop_a,
      UnaryTPP  *argop_b,
      UnaryTPP  *argop_c,
      BinaryTPP *postop)
      : BrgemmExtConvTPP(
            M,
            N,
            K,
            str_a,
            str_b,
            lda,
            ldb,
            ldc,
            1.0 /*beta */,
            0 /*a_trans */,
            0 /*unroll_hint */,
            argop_a,
            argop_b,
            argop_c,
            postop) {}
  BrgemmExtConvTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      long lda,
      long ldb,
      long ldc,
      float beta,
      int a_trans,
      int unroll_hint,
      UnaryTPP  *argop_a,
      UnaryTPP  *argop_b,
      UnaryTPP  *argop_c,
      BinaryTPP *postop)
      : BrgemmExtConvTPP(
            M,
            N,
            K,
            str_a,
            str_b,
            (a_trans == 0 ? K : M),
            N,
            N,
            beta,
            a_trans,
            0 /*c_vnni*/,
            unroll_hint,
            argop_a,
            argop_b,
            argop_c,
            postop) {}
  BrgemmExtConvTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      long lda,
      long ldb,
      long ldc,
      float beta,
      int a_trans,
      int c_vnni,
      int unroll_hint,
      UnaryTPP  *argop_a,
      UnaryTPP  *argop_b,
      UnaryTPP  *argop_c,
      BinaryTPP *postop)
      : BrgemmBaseTPP<Tin,Tout>(M, N, K, false, str_a, str_b, lda, ldb, ldc, beta, a_trans, c_vnni, unroll_hint,
        argop_a, argop_b, argop_c, postop) {}

  void operator()(
      Tin* A,
      Tin* B,
      Tout* C,
      unsigned long long count,
      bool no_tile_cfg = false) {
    TPP_ASSERT(!has_postop, "Error: calling a brgemm with a postop but without an extra input argument for it\n");

    libxsmm_gemm_ext_param gemm_ext_param;
    memset(&gemm_ext_param, 0, sizeof(libxsmm_gemm_ext_param));
    gemm_ext_param.op.tertiary = &count;
    gemm_ext_param.c.primary = (void*)C;
    gemm_ext_param.a.primary = (void*)B;
    gemm_ext_param.b.primary = (void*)A;
    if (!no_tile_cfg) {
      k_ext_gemm_with_tc(&gemm_ext_param);
    } else {
      k_ext_gemm_no_tc(&gemm_ext_param);
    }
  }

  void operator()(
      Tin* A,
      Tin* B,
      Tout* C,
      Tout* D,
      unsigned long long count,
      bool no_tile_cfg = false) {
    TPP_ASSERT(has_postop, "Error: calling a brgemm without postop with an extra input argument\n");

    libxsmm_gemm_ext_param gemm_ext_param;
    memset(&gemm_ext_param, 0, sizeof(libxsmm_gemm_ext_param));
    gemm_ext_param.op.tertiary = &count;
    gemm_ext_param.c.primary = (void*)C;
    gemm_ext_param.a.primary = (void*)B;
    gemm_ext_param.b.primary = (void*)A;
    if (has_postop)
      gemm_ext_param.d.primary = (void*)D;
    if (!no_tile_cfg) {
      k_ext_gemm_with_tc(&gemm_ext_param);
    } else {
      k_ext_gemm_no_tc(&gemm_ext_param);
    }
  }

  void ref(
      Tin* A,
      Tin* B,
      Tout* C,
      unsigned long long count,
      bool no_tile_cfg = false) {
    TPP_ASSERT(false, "ref() has not been implemented for BrgemmExtConvTPP\n");
  }
  void ref(
      Tin* A,
      Tin* B,
      Tout* C,
      Tout* D,
      unsigned long long count,
      bool no_tile_cfg = false) {
    TPP_ASSERT(false, "ref() has not been implemented for BrgemmExtConvTPP\n");
  }
};

template <typename Tin, typename Tout>
class BrgemmOffsetTPP : public BrgemmBaseTPP<Tin,Tout> {
 using BrgemmBaseTPP<Tin,Tout>::M;
 using BrgemmBaseTPP<Tin,Tout>::N;
 using BrgemmBaseTPP<Tin,Tout>::K;
 using BrgemmBaseTPP<Tin,Tout>::lda;
 using BrgemmBaseTPP<Tin,Tout>::ldb;
 using BrgemmBaseTPP<Tin,Tout>::ldc;
 using BrgemmBaseTPP<Tin,Tout>::beta;
 using BrgemmBaseTPP<Tin,Tout>::k_gemm_with_tc;
 using BrgemmBaseTPP<Tin,Tout>::k_gemm_no_tc;
 public:
  BrgemmOffsetTPP() : BrgemmBaseTPP<Tin,Tout>() {}
  BrgemmOffsetTPP(
      long M,
      long N,
      long K,
      long lda,
      long ldb,
      long ldc,
      float beta,
      int a_trans,
      int unroll_hint)
      : BrgemmOffsetTPP(
            M,
            N,
            K,
            lda,
            ldb,
            ldc,
            beta,
            a_trans,
            false,
            unroll_hint) {}
  BrgemmOffsetTPP(
      long M,
      long N,
      long K,
      long lda,
      long ldb,
      long ldc,
      float beta,
      int a_trans,
      int c_vnni,
      int unroll_hint)
      : BrgemmBaseTPP<Tin,Tout>(M, N, K, true, 0, 0, lda, ldb, ldc, beta, a_trans, c_vnni, unroll_hint) {}
  void operator()(
      Tin* A,
      Tin* B,
      Tout* C,
      unsigned long long *A_offsets,
      unsigned long long *B_offsets,
      unsigned long long count,
      bool no_tile_cfg = false) {
    libxsmm_gemm_param gemm_param;
    memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
    gemm_param.op.tertiary = &count;
    gemm_param.c.primary = (void*)C;
    gemm_param.a.primary = (void*)B;
    gemm_param.a.secondary = (void*)B_offsets;
    gemm_param.b.primary = (void*)A;
    gemm_param.b.secondary = (void*)A_offsets;

    if (!no_tile_cfg) {
      k_gemm_with_tc(&gemm_param);
    } else {
      k_gemm_no_tc(&gemm_param);
    }
  }

  void ref(
      Tin* A,
      Tin* B,
      Tout* C,
      unsigned long long *A_offsets,
      unsigned long long *B_offsets,
      unsigned long long count,
      bool no_tile_cfg = false) {
    printf("ref() not implemented for BrgemmOffsetTPP\n");
    exit(-1);
  }
};


template <typename Tin, typename Tout>
class BrgemmOffsetExtConvTPP : public BrgemmBaseTPP<Tin,Tout> {
 using BrgemmBaseTPP<Tin,Tout>::M;
 using BrgemmBaseTPP<Tin,Tout>::N;
 using BrgemmBaseTPP<Tin,Tout>::K;
 using BrgemmBaseTPP<Tin,Tout>::lda;
 using BrgemmBaseTPP<Tin,Tout>::ldb;
 using BrgemmBaseTPP<Tin,Tout>::ldc;
 using BrgemmBaseTPP<Tin,Tout>::beta;
 using BrgemmBaseTPP<Tin,Tout>::a_trans;
 using BrgemmBaseTPP<Tin,Tout>::has_postop;
 using BrgemmBaseTPP<Tin,Tout>::k_ext_gemm_with_tc;
 using BrgemmBaseTPP<Tin,Tout>::k_ext_gemm_no_tc;
 public:
  BrgemmOffsetExtConvTPP() : BrgemmBaseTPP<Tin,Tout>() {}

  BrgemmOffsetExtConvTPP(
      long M,
      long N,
      long K,
      long lda,
      long ldb,
      long ldc,
      float beta,
      int a_trans,
      int unroll_hint,
      UnaryTPP  *argop_a,
      UnaryTPP  *argop_b,
      UnaryTPP  *argop_c,
      BinaryTPP *postop)
      : BrgemmOffsetExtConvTPP(
            M,
            N,
            K,
            lda,
            ldb,
            ldc,
            beta,
            a_trans,
            false,
            unroll_hint,
            argop_a,
            argop_b,
            argop_c,
            postop) {}
  BrgemmOffsetExtConvTPP(
      long M,
      long N,
      long K,
      long lda,
      long ldb,
      long ldc,
      float beta,
      int a_trans,
      int c_vnni,
      int unroll_hint,
      UnaryTPP  *argop_a,
      UnaryTPP  *argop_b,
      UnaryTPP  *argop_c,
      BinaryTPP *postop)
      : BrgemmBaseTPP<Tin,Tout>(M, N, K, true, 0, 0, lda, ldb, ldc, beta, a_trans, c_vnni, unroll_hint,
        argop_a, argop_b, argop_c, postop) {}

  void operator()(
      Tin* A,
      Tin* B,
      Tout* C,
      unsigned long long *A_offsets,
      unsigned long long *B_offsets,
      unsigned long long count,
      bool no_tile_cfg = false) {
    TPP_ASSERT(!has_postop, "Error: calling a brgemm (offset-based) with a postop but without an extra input argument for it\n");

    libxsmm_gemm_ext_param gemm_ext_param;
    memset(&gemm_ext_param, 0, sizeof(libxsmm_gemm_ext_param));
    gemm_ext_param.op.tertiary = &count;
    gemm_ext_param.c.primary = (void*)C;
    gemm_ext_param.a.primary = (void*)B;
    gemm_ext_param.b.primary = (void*)A;
    gemm_ext_param.a.secondary = (void*)B_offsets;
    gemm_ext_param.b.secondary = (void*)A_offsets;
    if (!no_tile_cfg) {
      k_ext_gemm_with_tc(&gemm_ext_param);
    } else {
      k_ext_gemm_no_tc(&gemm_ext_param);
    }
  }

  void operator()(
      Tin* A,
      Tin* B,
      Tout* C,
      Tout* D,
      unsigned long long *A_offsets,
      unsigned long long *B_offsets,
      unsigned long long count,
      bool no_tile_cfg = false) {
    TPP_ASSERT(has_postop, "Error: calling a brgemm (offset-based) without postop with an extra input argument\n");

    libxsmm_gemm_ext_param gemm_ext_param;
    memset(&gemm_ext_param, 0, sizeof(libxsmm_gemm_ext_param));
    gemm_ext_param.op.tertiary = &count;
    gemm_ext_param.c.primary = (void*)C;
    gemm_ext_param.a.primary = (void*)B;
    gemm_ext_param.b.primary = (void*)A;
    gemm_ext_param.a.secondary = (void*)B_offsets;
    gemm_ext_param.b.secondary = (void*)A_offsets;
    if (has_postop)
      gemm_ext_param.d.primary = (void*)D;
    if (!no_tile_cfg) {
      k_ext_gemm_with_tc(&gemm_ext_param);
    } else {
      k_ext_gemm_no_tc(&gemm_ext_param);
    }
  }

  void ref(
      Tin* A,
      Tin* B,
      Tout* C,
      unsigned long long *A_offsets,
      unsigned long long *B_offsets,
      unsigned long long count,
      bool no_tile_cfg = false) {
    TPP_ASSERT(false, "ref() has not been implemented for BrgemmOffsetExtConvTPP\n");
  }
  void ref(
      Tin* A,
      Tin* B,
      Tout* C,
      Tout* D,
      unsigned long long *A_offsets,
      unsigned long long *B_offsets,
      unsigned long long count,
      bool no_tile_cfg = false) {
    TPP_ASSERT(false, "ref() has not been implemented for BrgemmOffsetExtConvTPP\n");
  }
};




template <typename Tin, typename Tout = Tin>
class GeluFwdTPP {
 public:
  GeluFwdTPP() {}
  GeluFwdTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_GELU) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
#ifdef __AVX512F__
    int i;
    for (i = 0; i < ALIGNDOWN(N, 16); i += 16) {
      auto vin = _mm512_loadu_ps_auto(&in[i]);
      // auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
      auto vout = LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(vin);
      _mm512_storeu_ps_auto(&out[i], vout);
    }
    if (i < N) {
      int rem = N - i;
      __mmask16 mask = (1 << rem) - 1;
      auto vin = _mm512_maskz_loadu_ps_auto(mask, &in[i]);
      // auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
      auto vout = LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(vin);
      _mm512_mask_storeu_ps_auto(&out[i], mask, vout);
    }
#else
    for (int i = 0; i < N; i++) {
      float x = in[i];
      out[i] = (erff(x / sqrtf(2.0)) + 1.0) * 0.5 * x;
    }
#endif
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename T1, typename T2 = T1, typename T3 = T1>
class GeluBwdTPP : public BaseTPP {
 public:
  GeluBwdTPP() {}
  GeluBwdTPP(int N) : N(N) {
    kernel = (libxsmm_matrix_eqn_function)get_kernel();
    initialized = true;
  }
  void operator()(T1* gout, T2* in, T3* gin) {
    if (!initialized)
      return;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[2];
    arg_array[0].primary = (void*)gout;
    arg_array[1].primary = (void*)in;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)gin;

    kernel(&eqn_param);
  }
  void ref(T1* gout, T2* in, T3* gin) {
#ifdef __AVX512F__
    int i;
    for (i = 0; i < ALIGNDOWN(N, 16); i += 16) {
      auto vgout = _mm512_loadu_ps_auto(&gout[i]);
      auto vin_gelu = _mm512_loadu_ps_auto(&in[i]);
      auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin_gelu);
      // auto vgin_gelu =
      // LIBXSMM_INTRINSICS_MM512_GELU_BWD_PS_MINIMAX3(vin_gelu);
      auto vout = _mm512_mul_ps(vgin_gelu, vgout);
      _mm512_storeu_ps_auto(&gin[i], vout);
    }
    if (i < N) {
      int rem = N - i;
      __mmask16 mask = (1 << rem) - 1;
      auto vgout = _mm512_maskz_loadu_ps_auto(mask, &gout[i]);
      auto vin_gelu = _mm512_maskz_loadu_ps_auto(mask, &in[i]);
      auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin_gelu);
      // auto vgin_gelu =
      // LIBXSMM_INTRINSICS_MM512_GELU_BWD_PS_MINIMAX3(vin_gelu);
      auto vout = _mm512_mul_ps(vgin_gelu, vgout);
      _mm512_mask_storeu_ps_auto(&gin[i], mask, vout);
    }
#else
    constexpr float PI = 3.14159265358979323846;
    for (int i = 0; i < N; i++) {
      float x = in[i];
      gin[i] = (float)gout[i] *
          (0.5 + 0.5 * erff(x / sqrtf(2.0)) +
           x / (sqrtf(2.0 * PI)) * expf(-0.5 * x * x));
    }
#endif
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(
        hash,
        200,
        "gelu_bwd_eqn_t%d_%d_%d_i%d",
        XsmmDtype<T1>(),
        XsmmDtype<T2>(),
        XsmmDtype<T3>(),
        N);
    return std::string(hash);
  }
  void* build_kernel() override {
    auto dt1 = XsmmDtype<T1>();
    auto dt2 = XsmmDtype<T2>();
    auto dt3 = XsmmDtype<T3>();
    libxsmm_blasint ld = N;
    libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL);
    meqn_push_arg(my_eqn0, N, 1, N, 0, 0, dt1);
    meqn_push_unary_op(my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_GELU_INV);
    meqn_push_arg(my_eqn0, N, 1, N, 1, 0, dt2);
    debug_print_eqn_tree(my_eqn0);
    return (void*)meqn_dispatch(N, 1, &ld, dt3, my_eqn0);
  }

 private:
  int N = 0;
  libxsmm_matrix_eqn_function kernel = NULL;
};

template <typename Tin, typename Tout = Tin>
class ReLUFwdTPP {
 public:
  ReLUFwdTPP() {}
  ReLUFwdTPP(int N, bool bm) : ReLUFwdTPP(1, N, bm) {}
  ReLUFwdTPP(int rows, int cols, bool bm)
      : ReLUFwdTPP(rows, cols, cols, cols, bm) {}
  ReLUFwdTPP(int rows, int cols, int ldi, int ldo, bool bm)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
#ifdef __x86_64__
            (XsmmDtype<Tin>() == LIBXSMM_DATATYPE_BF16 ? LIBXSMM_DATATYPE_BF16 : LIBXSMM_DATATYPE_F32),
#else
            LIBXSMM_DATATYPE_F32,
#endif
            bm ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT
               : LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RELU) {}
  void operator()(Tin* in, Tout* out, short* mask = NULL) {
    kernel((void*)in, (void*)out, (void*)mask);
  }
  void ref(Tin* in, Tout* out, short* mask = NULL) {
    kernel((void*)in, (void*)out, (void*)mask);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class ReLUFwdConvTPP : public UnaryTPP {
 public:
  ReLUFwdConvTPP() : UnaryTPP() {}
  ReLUFwdConvTPP(int N, bool bm) : ReLUFwdConvTPP(1, N, bm) {}
  ReLUFwdConvTPP(int rows, int cols, bool bm)
      : ReLUFwdConvTPP(rows, cols, cols, cols, bm) {}
  ReLUFwdConvTPP(int rows, int cols, int ldi, int ldo, bool bm)
      : UnaryTPP(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
#ifdef __x86_64__
            (XsmmDtype<Tin>() == LIBXSMM_DATATYPE_BF16 ? LIBXSMM_DATATYPE_BF16 : LIBXSMM_DATATYPE_F32),
#else
            LIBXSMM_DATATYPE_F32,
#endif
            bm ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT
               : LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RELU) {}
  void operator()(Tin* in, Tout* out, short* mask = NULL) {
    UnaryTPP::operator()((void*)in, (void*)out, (void*)mask);
  }
  void ref(Tin* in, Tout* out, short* mask = NULL) {
    UnaryTPP::operator()((void*)in, (void*)out, (void*)mask);
  }
};


template <typename Tin, typename Tout = Tin>
class ReLUBwdTPP {
 public:
  ReLUBwdTPP() {}
  ReLUBwdTPP(int N, bool bm) : ReLUBwdTPP(1, N, bm) {}
  ReLUBwdTPP(int rows, int cols, bool bm)
      : ReLUBwdTPP(rows, cols, cols, cols, bm) {}
  ReLUBwdTPP(int rows, int cols, int ldi, int ldo, bool bm)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        bm(bm),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            bm ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT
               : LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) {}
  void operator()(Tin* in, Tout* out, Tin* in2 = NULL, short* mask = NULL) {
    kernel(in, bm ? (void*)mask : (void*)in2, NULL, out, NULL);
  }
  void ref(Tin* in, Tout* out, Tin* in2 = NULL, short* mask = NULL) {
    kernel(in, bm ? (void*)mask : (void*)in2, NULL, out, NULL);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  bool bm;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class ELUFwdTPP {
 public:
  ELUFwdTPP() {}
  ELUFwdTPP(int N, float alpha) : ELUFwdTPP(1, N, alpha) {}
  ELUFwdTPP(int rows, int cols, float alpha)
      : ELUFwdTPP(rows, cols, cols, cols, alpha) {}
  ELUFwdTPP(int rows, int cols, int ldi, int ldo, float alpha)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        alpha(alpha),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_ELU) {}
  void operator()(Tin* in, Tout* out) {
    kernel(in, NULL, NULL, &alpha, NULL, NULL, out, NULL);
  }
  void ref(Tin* in, Tout* out) {
    Tin a = alpha;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        out[i * ldo + j] = in[i * ldi + j] > 0 ? in[i * ldi + j]
                                               : a * (exp(in[i * ldi + j]) - 1);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float alpha;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class ELUBwdTPP {
 public:
  ELUBwdTPP() {}
  ELUBwdTPP(int N, float alpha) : ELUBwdTPP(1, N, alpha) {}
  ELUBwdTPP(int rows, int cols, float alpha)
      : ELUBwdTPP(rows, cols, cols, cols, alpha) {}
  ELUBwdTPP(int rows, int cols, int ldi, int ldo, float alpha)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        alpha(alpha),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) {}
  void operator()(Tin* in, Tin* in2, Tout* out) {
    kernel(in, in2, NULL, &alpha, NULL, NULL, out, NULL);
  }
  void ref(Tin* in, Tin* in2, Tout* out) {
    Tin a = alpha;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        out[i * ldo + j] = in2[i * ldi + j] > 0
            ? in[i * ldi + j]
            : in[i * ldi + j] * in2[i * ldi + j] + a * in[i * ldi + j];
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float alpha;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class LeakyReLUFwdTPP {
 public:
  LeakyReLUFwdTPP() {}
  LeakyReLUFwdTPP(int N, float alpha) : LeakyReLUFwdTPP(1, N, alpha) {}
  LeakyReLUFwdTPP(int rows, int cols, float alpha)
      : LeakyReLUFwdTPP(rows, cols, cols, cols, alpha) {}
  LeakyReLUFwdTPP(int rows, int cols, int ldi, int ldo, float alpha)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        alpha(alpha),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT,
            LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) {}
  void operator()(Tin* in, Tout* out, short* mask = NULL) {
    kernel(in, NULL, NULL, &alpha, NULL, NULL, out, mask);
  }
  void ref(Tin* in, Tout* out, short* mask = NULL) {
    float a = alpha;
    // std::cout << " op: " << out << " inp: "<< in << std::endl;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = in[i * ldi + j] > 0 ? (Tout)in[i * ldi + j]
                                               : (Tout)(a * (in[i * ldi + j]));
      }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float alpha;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class LeakyReLUBwdTPP {
 public:
  LeakyReLUBwdTPP() {}
  LeakyReLUBwdTPP(int N, float alpha) : LeakyReLUBwdTPP(1, N, alpha) {}
  LeakyReLUBwdTPP(int rows, int cols, float alpha)
      : LeakyReLUBwdTPP(rows, cols, cols, cols, alpha) {}
  LeakyReLUBwdTPP(int rows, int cols, int ldi, int ldo, float alpha)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        alpha(alpha),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT,
            LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) {}
  void operator()(Tin* in, Tout* out, Tin* in2 = NULL, short* mask = NULL) {
    kernel(in, mask, NULL, &alpha, NULL, NULL, out, NULL);
  }
  void ref(Tin* in, Tout* out, Tin* in2, short* mask = NULL) {
    float a = alpha;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) {
        float grad_out = in[i * ldi + j];
        out[i * ldo + j] =
            in2[i * ldi + j] > 0 ? (Tout)grad_out : (Tout)(a * grad_out);
      }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float alpha;
  UnaryTPP kernel;
};

template <typename T>
class SiLUFwdTPP {
 public:
  SiLUFwdTPP() {}
  SiLUFwdTPP(int N) : SiLUFwdTPP(1, N) {}
  SiLUFwdTPP(int rows, int cols) : SiLUFwdTPP(rows, cols, cols, cols) {}
  SiLUFwdTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        sigmoid(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_SIGMOID),
        mul(rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(T* in, T* out, T* sigout) {
    sigmoid((void*)in, (void*)sigout);
    mul((void*)in, (void*)sigout, (void*)out);
  }
  void ref(T* in, T* out, T* sigout) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        sigout[i * ldo + j] = 1. / (1. + exp(-in[i * ldi + j]));
        out[i * ldo + j] = in[i * ldi + j] * sigout[i * ldo + j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP sigmoid;
  BinaryTPP mul;
};

template <typename Tin, typename Tout = Tin>
class SiLUBwdTPP : public BaseTPP {
 public:
  SiLUBwdTPP() {}
  SiLUBwdTPP(int N) : SiLUBwdTPP(1, N) {}
  SiLUBwdTPP(int rows, int cols) : SiLUBwdTPP(rows, cols, cols, cols) {}
  SiLUBwdTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows), cols(cols), ldi(ldi), ldo(ldo) {
    kernel = (libxsmm_matrix_eqn_function)get_kernel();
    initialized = true;
  }
  void operator()(Tin* in, Tin* in2, Tin* in3, Tout* out) {
    if (!initialized)
      return;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[5];
    float one = 1.;
    arg_array[0].primary = (void*)in;
    arg_array[1].primary = (void*)in2;
    arg_array[2].primary = (void*)in3;
    arg_array[3].primary = (void*)&one;
    arg_array[4].primary = (void*)in2;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)out;

    kernel(&eqn_param);
  }
  void ref(Tin* in, Tin* in2, Tin* in3, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        float grad_out = in[i * ldi + j];
        float si = in2[i * ldi + j];
        float fout = in3[i * ldi + j];

        out[i] = grad_out * (si + fout * (1 - si));
      }
    }
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(hash, 200, "silu_bwd_eqn_%d_%d", rows, cols);
    return std::string(hash);
  }
  void* build_kernel() override {
    libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 0, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 1, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 2, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_BINARY_SUB,
        LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 3, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 4, 0, LIBXSMM_DATATYPE_F32);

    auto func0 = meqn_dispatch(cols, rows, &ldo, XsmmDtype<Tout>(), my_eqn0);
    return (void*)func0;
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  libxsmm_matrix_eqn_function kernel = NULL;
};

template <typename Tin, typename Tout = Tin>
class DropOutFwdTPP {
 public:
  DropOutFwdTPP() {}
  DropOutFwdTPP(int N, float p) : DropOutFwdTPP(1, N, p) {}
  DropOutFwdTPP(int rows, int cols, float p)
      : DropOutFwdTPP(rows, cols, cols, cols, p) {}
  DropOutFwdTPP(int rows, int cols, int ldi, int ldo, float p)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        p(p),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT,
            LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) {}
  void operator()(Tin* in, void* rng_state, Tout* out, short* mask) {
    kernel(in, NULL, NULL, &p, rng_state, NULL, out, mask);
  }
  void ref(Tin* in, void* rng_state, Tout* out, short* mask) {
    kernel(in, NULL, NULL, &p, rng_state, NULL, out, mask);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float p;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class DropOutBwdTPP {
 public:
  DropOutBwdTPP() {}
  DropOutBwdTPP(int N, float p) : DropOutBwdTPP(1, N, p) {}
  DropOutBwdTPP(int rows, int cols, float p)
      : DropOutBwdTPP(rows, cols, cols, cols, p) {}
  DropOutBwdTPP(int rows, int cols, int ldi, int ldo, float p)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        p(p),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT,
            LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) {}
  void operator()(Tin* in, Tout* out, short* mask) {
    kernel(in, mask, NULL, &p, NULL, NULL, out, NULL);
  }
  void ref(Tin* in, Tout* out, short* mask) {
    kernel(in, mask, NULL, &p, NULL, NULL, out, NULL);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float p;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout>
class SoftMaxFwdTPP {
 public:
  SoftMaxFwdTPP() {}
  SoftMaxFwdTPP(int S1, int S2, int S3)
      : S1(S1), S2(S2), S3(S3), eqn0(S1, S2, S3), eqn1(S1, S2, S3) {}
  void operator()(Tin* in, Tout* out) {
    LIBXSMM_ALIGNED(float tmp[S1 * S3], 64);
    for (int s2 = 0; s2 < S2; s2++) {
      eqn0(&in[s2 * S3], tmp);
      eqn1(tmp, &out[s2 * S3]);
    }
  }
  void ref(Tin* pinp, Tout* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, Tin, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, Tout, out, pout, S2, S3);
#if defined(__AVX512F__)
    for (s2 = 0; s2 < S2; s2++) {
      float tmp[S1][S3];
      float max =
          upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
      float sum = 0.0;
      __m512 vmax = _mm512_set1_ps(max);
      __m512 vsum = _mm512_setzero_ps();

      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          vmax = _mm512_max_ps(
              _mm512_loadu_ps_auto(
                  &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          vmax = _mm512_mask_max_ps(
              vmax,
              mask,
              _mm512_maskz_loadu_ps_auto(
                  mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax);
        }
      }
      max = _mm512_reduce_max_ps(vmax);
      vmax = _mm512_set1_ps(max);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(
              _mm512_loadu_ps_auto(
                  &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax));
          _mm512_storeu_ps(&tmp[s1][s3], vz);
          vsum = _mm512_add_ps(vsum, vz);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(
              _mm512_maskz_loadu_ps_auto(
                  mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax));
          _mm512_mask_storeu_ps(&tmp[s1][s3], mask, vz);
          vsum = _mm512_mask_add_ps(vsum, mask, vsum, vz);
        }
      }
      sum = _mm512_reduce_add_ps(vsum);
      sum = 1.0 / sum;
      vsum = _mm512_set1_ps(sum);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          _mm512_storeu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
              _mm512_mul_ps(vsum, _mm512_loadu_ps(&tmp[s1][s3])));
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          _mm512_mask_storeu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
              mask,
              _mm512_mul_ps(vsum, _mm512_maskz_loadu_ps(mask, &tmp[s1][s3])));
        }
      }
    }
#else
    for (s2 = 0; s2 < S2; s2++) {
      float tmp[S1][S3];
      float max =
          upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
      float sum = 0.0;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = upconvert_to_float(
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
          if (max < cur)
            max = cur;
        }
      }
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = upconvert_to_float(
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
          float z = expf(cur - max);
          tmp[s1][s3] = z;
          sum += z;
        }
      }
      sum = 1.0 / sum;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = tmp[s1][s3] * sum;
          // libxsmm_rne_convert_fp32_bf16( &cur, &LIBXSMM_VLA_ACCESS(3, out,
          // s1, s2, s3, S2, S3), 1 );
          LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) = cur;
        }
      }
    }
#endif
  }
  class Eqn0 : BaseTPP {
   public:
    Eqn0() {}
    Eqn0(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(Tin* in, float* out) {
      if (!initialized)
        return;
      libxsmm_matrix_eqn_param eqn_param;
      libxsmm_matrix_arg arg_array[1];
      arg_array[0].primary = (void*)in;
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = (void*)out;

      kernel(&eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "softmax_fwd_eqn0_ti%d_to%d_S1%d_S2%d_S3%d",
          XsmmDtype<Tin>(),
          LIBXSMM_DATATYPE_F32,
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto dt_in = XsmmDtype<Tin>();
      libxsmm_blasint tmp_ld = S2;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
      meqn_push_unary_op(my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_EXP);
      meqn_push_binary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_BINARY_SUB,
          LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, dt_in);
      meqn_push_unary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX,
          LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
      meqn_push_unary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX,
          LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, dt_in);
      debug_print_eqn_tree(my_eqn0); // printf
      return (void*)meqn_dispatch(
          S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn0);
    }

   private:
    int S1, S2, S3;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

  class Eqn1 : BaseTPP {
   public:
    Eqn1() {}
    Eqn1(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(float* in, Tout* out) {
      if (!initialized)
        return;
      libxsmm_matrix_eqn_param eqn_param;
      libxsmm_matrix_arg arg_array[1];
      arg_array[0].primary = (void*)in;
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = (void*)out;

      kernel(&eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "softmax_fwd_eqn1_ti%d_to%d_S1%d_S2%d_S3%d",
          LIBXSMM_DATATYPE_F32,
          XsmmDtype<Tout>(),
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto dt_out = XsmmDtype<Tout>();
      libxsmm_blasint tmp_ld = S2;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn1 = libxsmm_matrix_eqn_create();
      meqn_push_binary_op(
          my_eqn1,
          LIBXSMM_MELTW_TYPE_BINARY_MUL,
          LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
      meqn_push_arg(my_eqn1, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_unary_op(my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL);
      meqn_push_unary_op(
          my_eqn1,
          LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
          LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
      meqn_push_unary_op(
          my_eqn1,
          LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
          LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
      meqn_push_arg(my_eqn1, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
      /*debug_print_eqn_tree( my_eqn1 );*/
      return (void*)meqn_dispatch(S3, S1, &ld, dt_out, my_eqn1);
    }

   private:
    int S1, S2, S3;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  Eqn0 eqn0;
  Eqn1 eqn1;
};

template <typename T1, typename T2, typename T3>
class SoftMaxBwdTPP {
 public:
  SoftMaxBwdTPP() {}
  SoftMaxBwdTPP(int S1, int S2, int S3)
      : S1(S1), S2(S2), S3(S3), eqn0(S1, S2, S3, 0), eqn1(S1, S2, S3, 1) {}
  void operator()(T1* gin, T2* gout, T3* out) {
    LIBXSMM_ALIGNED(float tmp[S1 * S3], 64);
    for (int s2 = 0; s2 < S2; s2++) {
      libxsmm_matrix_eqn_param eqn_param;
      libxsmm_matrix_arg arg_array[2];
      arg_array[0].primary = (void*)&gout[s2 * S3];
      arg_array[1].primary = (void*)&out[s2 * S3];
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = (void*)tmp;
      eqn0(&eqn_param);

      arg_array[0].primary = (void*)tmp;
      eqn_param.output.primary = (void*)&gin[s2 * S3];
      eqn1(&eqn_param);
    }
  }
  void ref(T1* pgradinp, T2* pgradout, T3* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T1, ginp, pgradinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T2, gout, pgradout, S2, S3);
    LIBXSMM_VLA_DECL(3, T3, out, pout, S2, S3);
#if defined(__AVX512F__)
    for (s2 = 0; s2 < S2; s2++) {
      float sum = 0.0;
      __m512 vsum = _mm512_setzero_ps();
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 vgo =
              _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
          __m512 vo = _mm512_loadu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
          vsum = _mm512_fmadd_ps(vgo, vo, vsum);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vgo = _mm512_maskz_loadu_ps(
              mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
          __m512 vo = _mm512_maskz_loadu_ps_auto(
              mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
          vsum = _mm512_fmadd_ps(vgo, vo, vsum);
        }
      }
      sum = _mm512_reduce_add_ps(vsum);
      vsum = _mm512_set1_ps(sum);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 tmp = _mm512_sub_ps(
              _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)),
              vsum);
          _mm512_storeu_ps(
              &LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3),
              _mm512_mul_ps(
                  _mm512_loadu_ps_auto(
                      &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)),
                  tmp));
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 tmp = _mm512_sub_ps(
              _mm512_maskz_loadu_ps(
                  mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)),
              vsum);
          _mm512_mask_storeu_ps(
              &LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3),
              mask,
              _mm512_mul_ps(
                  _mm512_maskz_loadu_ps_auto(
                      mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)),
                  tmp));
        }
      }
    }
#else
    for (s2 = 0; s2 < S2; s2++) {
      float sum = 0.0;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          sum += LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) *
              upconvert_to_float(
                     LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
        }
      }
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3) =
              upconvert_to_float(
                  LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)) *
              (LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) - sum);
        }
      }
    }
#endif
  }

  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3, int eqn_no)
        : S1(S1), S2(S2), S3(S3), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "softmax_bwd_eqn%d_t1%d_t2%d_t3%d_S1%d_S2%d_S3%d",
          eqn_no,
          XsmmDtype<T2>(),
          XsmmDtype<T3>(),
          LIBXSMM_DATATYPE_F32,
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto dt_1 = XsmmDtype<T1>();
      auto dt_2 = XsmmDtype<T2>();
      auto dt_3 = XsmmDtype<T3>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_matrix_eqn_function func;
      if (eqn_no == 0) {
        libxsmm_blasint my_eqn2 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_arg(my_eqn2, S3, S1, ld, 0, 0, dt_2);
        meqn_push_arg(my_eqn2, S3, S1, ld, 1, 0, dt_3);
        debug_print_eqn_tree(my_eqn2); // printf
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2);
      } else if (eqn_no == 1) {
        libxsmm_blasint my_eqn3 = libxsmm_matrix_eqn_create();
#if 1
        meqn_push_ternary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_TERNARY_NMULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_unary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_unary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn3, S3, S1, ld, 1, 0, dt_3);
#else
        meqn_push_binary_op(my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_SUB);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_binary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn3, S3, S1, ld, 1, 0, dt_3);
        meqn_push_unary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_unary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
#endif
        debug_print_eqn_tree(my_eqn3);
        func = meqn_dispatch(S3, S1, &ld, dt_1, my_eqn3);
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    int S1, S2, S3, eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  Eqn eqn0, eqn1;
};

template <typename Tin, typename Tout>
class VarSoftMaxFwdTPP {
 public:
  VarSoftMaxFwdTPP() {}
  VarSoftMaxFwdTPP(int S2, int S3)
      : S2(S2),
        S3(S3),
        kmax(
            1,
            S3,
            S3,
            S3,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX),
        ksub(
            1,
            S3,
            S3,
            S3,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1,
            LIBXSMM_MELTW_TYPE_BINARY_SUB),
        kexp(
            1,
            S3,
            S3,
            S3,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_EXP),
        ksum(
            1,
            S3,
            S3,
            S3,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        kmul(
            1,
            S3,
            S3,
            S3,
            LIBXSMM_DATATYPE_F32,
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(int S1, Tin* in, Tout* out) {
    LIBXSMM_ALIGNED(float tmp[S1 * S3], 64);
    for (int s2 = 0; s2 < S2; s2++) {
      Tin max = in[s2 * S3];
      float sum = 0.0f;
      for (int s1 = 0; s1 < S1; s1++) {
        float rmax = 0;
        kmax(&in[s1 * S2 * S3 + s2 * S3], &rmax);
        if (max < rmax)
          max = rmax;
      }
      for (int s1 = 0; s1 < S1; s1++) {
        LIBXSMM_ALIGNED(float tmp2[S3], 64);
        ksub(&in[s1 * S2 * S3 + s2 * S3], &max, tmp2);
        kexp(tmp2, &tmp[s1 * S3]);
        float lsum;
        ksum(&tmp[s1 * S3], &lsum);
        sum += lsum;
      }
      sum = 1.0 / sum;
      for (int s1 = 0; s1 < S1; s1++) {
        kmul(&tmp[s1 * S3], &sum, &out[s1 * S2 * S3 + s2 * S3]);
      }
    }
  }
  void ref(int S1, Tin* pinp, Tout* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, Tin, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, Tout, out, pout, S2, S3);
#if defined(__AVX512F__)
    for (s2 = 0; s2 < S2; s2++) {
      float tmp[S1][S3];
      float max =
          upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
      float sum = 0.0;
      __m512 vmax = _mm512_set1_ps(max);
      __m512 vsum = _mm512_setzero_ps();

      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          vmax = _mm512_max_ps(
              _mm512_loadu_ps_auto(
                  &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          vmax = _mm512_mask_max_ps(
              vmax,
              mask,
              _mm512_maskz_loadu_ps_auto(
                  mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax);
        }
      }
      max = _mm512_reduce_max_ps(vmax);
      vmax = _mm512_set1_ps(max);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(
              _mm512_loadu_ps_auto(
                  &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax));
          _mm512_storeu_ps(&tmp[s1][s3], vz);
          vsum = _mm512_add_ps(vsum, vz);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(
              _mm512_maskz_loadu_ps_auto(
                  mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax));
          _mm512_mask_storeu_ps(&tmp[s1][s3], mask, vz);
          vsum = _mm512_mask_add_ps(vsum, mask, vsum, vz);
        }
      }
      sum = _mm512_reduce_add_ps(vsum);
      sum = 1.0 / sum;
      vsum = _mm512_set1_ps(sum);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          _mm512_storeu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
              _mm512_mul_ps(vsum, _mm512_loadu_ps(&tmp[s1][s3])));
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          _mm512_mask_storeu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
              mask,
              _mm512_mul_ps(vsum, _mm512_maskz_loadu_ps(mask, &tmp[s1][s3])));
        }
      }
    }
#else
    //#warning "Not using AVX512 path for VarSoftMax"
    for (s2 = 0; s2 < S2; s2++) {
      float tmp[S1][S3];
      float max =
          upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
      float sum = 0.0;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = upconvert_to_float(
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
          if (max < cur)
            max = cur;
        }
      }
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = upconvert_to_float(
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
          float z = expf(cur - max);
          tmp[s1][s3] = z;
          sum += z;
        }
      }
      sum = 1.0 / sum;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = tmp[s1][s3] * sum;
          // libxsmm_rne_convert_fp32_bf16( &cur, &LIBXSMM_VLA_ACCESS(3, out,
          // s1, s2, s3, S2, S3), 1 );
          LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) = cur;
        }
      }
    }
#endif
  }

 private:
  int S2, S3;
  UnaryTPP kmax;
  BinaryTPP ksub;
  UnaryTPP kexp;
  UnaryTPP ksum;
  BinaryTPP kmul;
};

template <typename T1, typename T2, typename T3>
class VarSoftMaxBwdTPP {
 public:
  VarSoftMaxBwdTPP() {}
  VarSoftMaxBwdTPP(int S2, int S3) : S2(S2), S3(S3), eqn0(S3, 0), eqn1(S3, 1) {}
  void operator()(int S1, T1* gin, T2* gout, T3* out) {
    long S23 = S2 * S3;
    for (int s2 = 0; s2 < S2; s2++) {
      float tmp = 0.0f;
      libxsmm_matrix_eqn_param eqn_param;
      libxsmm_matrix_arg arg_array[3];
      arg_array[2].primary = (void*)&tmp;
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = (void*)&tmp;
      for (int s1 = 0; s1 < S1; s1++) {
        long ind = s1 * S23 + s2 * S3;
        arg_array[0].primary = (void*)&gout[ind];
        arg_array[1].primary = (void*)&out[ind];
        eqn0(&eqn_param);
      }
      for (int s1 = 0; s1 < S1; s1++) {
        long ind = s1 * S23 + s2 * S3;
        arg_array[0].primary = (void*)&gout[ind];
        arg_array[1].primary = (void*)&out[ind];
        eqn_param.output.primary = (void*)&gin[ind];
        eqn1(&eqn_param);
      }
    }
  }
  void ref(int S1, T1* pgradinp, T2* pgradout, T3* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T1, ginp, pgradinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T2, gout, pgradout, S2, S3);
    LIBXSMM_VLA_DECL(3, T3, out, pout, S2, S3);
#if defined(__AVX512F__)
    for (s2 = 0; s2 < S2; s2++) {
      float sum = 0.0;
      __m512 vsum = _mm512_setzero_ps();
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 vgo =
              _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
          __m512 vo = _mm512_loadu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
          vsum = _mm512_fmadd_ps(vgo, vo, vsum);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vgo = _mm512_maskz_loadu_ps(
              mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
          __m512 vo = _mm512_maskz_loadu_ps_auto(
              mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
          vsum = _mm512_fmadd_ps(vgo, vo, vsum);
        }
      }
      sum = _mm512_reduce_add_ps(vsum);
      vsum = _mm512_set1_ps(sum);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 tmp = _mm512_sub_ps(
              _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)),
              vsum);
          _mm512_storeu_ps(
              &LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3),
              _mm512_mul_ps(
                  _mm512_loadu_ps_auto(
                      &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)),
                  tmp));
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 tmp = _mm512_sub_ps(
              _mm512_maskz_loadu_ps(
                  mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)),
              vsum);
          _mm512_mask_storeu_ps(
              &LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3),
              mask,
              _mm512_mul_ps(
                  _mm512_maskz_loadu_ps_auto(
                      mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)),
                  tmp));
        }
      }
    }
#else
    for (s2 = 0; s2 < S2; s2++) {
      float sum = 0.0;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          sum += LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) *
              upconvert_to_float(
                     LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
        }
      }
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3) =
              upconvert_to_float(
                  LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)) *
              (LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) - sum);
        }
      }
    }
#endif
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S3, int eqn_no) : S3(S3), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "varsoftmax_bwd_eqn%d_t1%d_t2%d_t3%d_S3%d",
          eqn_no,
          XsmmDtype<T1>(),
          XsmmDtype<T2>(),
          XsmmDtype<T3>(),
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto dt_1 = XsmmDtype<T1>();
      auto dt_2 = XsmmDtype<T2>();
      auto dt_3 = XsmmDtype<T3>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint ld = S3;
      libxsmm_matrix_eqn_function func;
      if (eqn_no == 0) {
        libxsmm_blasint my_eqn2 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_arg(my_eqn2, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_unary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_arg(my_eqn2, S3, 1, ld, 0, 0, dt_2);
        meqn_push_arg(my_eqn2, S3, 1, ld, 1, 0, dt_3);
        debug_print_eqn_tree(my_eqn2); // printf
        func = meqn_dispatch(S3, 1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2);
      } else if (eqn_no == 1) {
        libxsmm_blasint my_eqn3 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_arg(my_eqn3, S3, 1, ld, 1, 0, dt_3);
        meqn_push_binary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_BINARY_SUB,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn3, S3, 1, ld, 0, 0, dt_2);
        meqn_push_arg(my_eqn3, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        debug_print_eqn_tree(my_eqn3);
        func = meqn_dispatch(S3, 1, &ld, dt_1, my_eqn3);
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    int S3, eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S2, S3;
  Eqn eqn0, eqn1;
};

template <typename T, typename LT = T>
class LayerNormFwdTPP {
 public:
  LayerNormFwdTPP() {}
  LayerNormFwdTPP(int S1, int S2, int S3, float eps)
      : S1(S1),
        S2(S2),
        S3(S3),
        eps(eps),
        reduce_cols_kernel(
            S1,
            S3,
            S2 * S3,
            S3,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD),
        reduce_rows_kernel(
            1,
            S3,
            S3,
            1,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        eqn(S1, S2, S3) {}
  void operator()(
      T* inp,
      LT* gamma,
      LT* beta,
      float* mean,
      float* var,
      T* out) {
    LIBXSMM_ALIGNED(float tmp[2 * S3], 64);
    const float c = 1.0 / ((float)S1 * S3);
    float m, v, s, b;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[5];
    eqn_param.inputs = arg_array;
    arg_array[1].primary = &s;
    arg_array[2].primary = &b;
    arg_array[3].primary = (void*)gamma;
    arg_array[4].primary = (void*)beta;
    for (int s2 = 0; s2 < S2; s2++) {
      reduce_cols_kernel((void*)&inp[s2 * S3], (void*)tmp);
      reduce_rows_kernel((void*)tmp, (void*)&m);
      reduce_rows_kernel((void*)&tmp[S3], (void*)&v);
      m = m * c;
      v = v * c;
      v = LIBXSMM_MAX(v - m * m, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      mean[s2] = m;
      var[s2] = v;
      s = v;
      b = -1.0 * v * m;
      arg_array[0].primary = (void*)&inp[s2 * S3];
      eqn_param.output.primary = (void*)&out[s2 * S3];
      eqn(&eqn_param);
    }
  }
  void ref(T* pinp, LT* pgamma, LT* pbeta, float* mean, float* var, T* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, out, pout, S2, S3);
    LIBXSMM_VLA_DECL(2, LT, gamma, pgamma, S3);
    LIBXSMM_VLA_DECL(2, LT, beta, pbeta, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float m = 0;
      float v = 0;
      float c = 1.0 / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          m += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
          v += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
        }
      }
      m = m * c;
      v = v * c;
      v = LIBXSMM_MAX(v - m * m, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      mean[s2] = m;
      var[s2] = v;
      float s = v;
      float b = -1.0 * v * m;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) =
              (LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) * s + b) *
                  LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) +
              LIBXSMM_VLA_ACCESS(2, beta, s1, s3, S3);
        }
      }
    }
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "layernorm_fwd_eqn_t1%d_t2%d_S1%d_S2%d_S3%d",
          XsmmDtype<T>(),
          XsmmDtype<LT>(),
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      auto bg_dt = XsmmDtype<LT>();
      auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = 1;
      libxsmm_blasint tmp_ld2 = S3;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
      meqn_push_ternary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
          LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
      meqn_push_ternary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
              LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
              LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, in_dt);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, S3, S1, tmp_ld2, 3, 0, bg_dt);
      meqn_push_arg(my_eqn0, S3, S1, tmp_ld2, 4, 0, bg_dt);
      debug_print_eqn_tree(my_eqn0); // printf
      return (void*)meqn_dispatch(S3, S1, &ld, out_dt, my_eqn0);
    }

   private:
    int S1, S2, S3;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  float eps;
  UnaryTPP reduce_cols_kernel;
  UnaryTPP reduce_rows_kernel;
  Eqn eqn;
};


template <typename T, typename LT = T>
class LayerNormBwdTPP {
 public:
  LayerNormBwdTPP() {}
  LayerNormBwdTPP(int S1, int S2, int S3)
      : S1(S1),
        S2(S2),
        S3(S3),
        dgamma_func(S1, S2, S3, 1),
        dbeta_func(S1, S2, S3, 2),
        db_func(S1, S2, S3, 3),
        ds_func(S1, S2, S3, 4),
        din_func(S1, S2, S3, 5) {}
  void operator()(
      T* dout,
      T* inp,
      float* mean,
      float* var,
      LT* gamma,
      T* din,
      float* dgamma,
      float* dbeta) {
    float a, b, c, db, ds;
    const float scale = 1.0f / ((float)S1 * S3);
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[8];
    eqn_param.inputs = arg_array;

    arg_array[1].primary = &a;
    arg_array[2].primary = &b;
    arg_array[4].primary = (void*)dgamma;
    arg_array[5].primary = (void*)dbeta;
    arg_array[6].primary = (void*)gamma;
    arg_array[7].primary = &c;

    for (int s2 = 0; s2 < S2; s2++) {
      a = var[s2];
      b = -a * mean[s2];
      arg_array[0].primary = (void*)&inp[s2 * S3];
      arg_array[3].primary = (void*)&dout[s2 * S3];

      eqn_param.output.primary = &ds;
      ds_func(&eqn_param);

      eqn_param.output.primary = &db;
      db_func(&eqn_param);

      eqn_param.output.primary = (void*)dgamma;
      dgamma_func(&eqn_param);

      eqn_param.output.primary = (void*)dbeta;
      dbeta_func(&eqn_param);

      b = (db * mean[s2] - ds) * a * a * a * scale;
      c = -b * mean[s2] - db * a * scale;

      eqn_param.output.primary = (void*)&din[s2 * S3];
      din_func(&eqn_param);
    }
  }
  void ref(
      T* pdout,
      T* pinp,
      float* mean,
      float* var,
      LT* pgamma,
      T* pdin,
      float* pdgamma,
      float* pdbeta) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, din, pdin, S2, S3);
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, dout, pdout, S2, S3);
    LIBXSMM_VLA_DECL(2, LT, gamma, pgamma, S3);
    LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, S3);
    LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float a = var[s2], c;
      float b = -a * mean[s2];
      float ds = 0.0f;
      float db = 0.0f;
      float scale = 1.0f / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3) +=
              (a * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + b) *
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
          LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3) +=
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
          ds += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
          db += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3);
        }
      }
      b = (db * mean[s2] - ds) * a * a * a * scale;
      c = -b * mean[s2] - db * a * scale;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3) =
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * a *
                  LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) +
              b * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + c;
        }
      }
    }
  }

  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3, int eqn_no)
        : S1(S1), S2(S2), S3(S3), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "layernorm_bwd_eqn%d_t1%d_t2%d_S1%d_S2%d_S3%d",
          eqn_no,
          XsmmDtype<T>(),
          XsmmDtype<LT>(),
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      auto bg_dt = XsmmDtype<LT>();
      // auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint tmp_ld2 = 1;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_matrix_eqn_function func = NULL;
      if (eqn_no == 1) {
        /* dgamma function  */
        libxsmm_blasint my_eqn1 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, S3, S1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn1, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn1, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn1, S3, S1, tmp_ld, 4, 0, LIBXSMM_DATATYPE_F32);
        /*debug_print_eqn_tree( my_eqn1 );*/
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn1);
      } else if (eqn_no == 2) {
        /* dbeta function  */
        libxsmm_blasint my_eqn2 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_arg(my_eqn2, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn2, S3, S1, tmp_ld, 5, 0, LIBXSMM_DATATYPE_F32);
        /*debug_print_eqn_tree( my_eqn1 );*/
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2);
      } else if (eqn_no == 3) {
        /* db equation */
        libxsmm_blasint my_eqn3 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(
            my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD);
        meqn_push_arg(my_eqn3, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 6, 0, bg_dt);
        func = meqn_dispatch(1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn3);
      } else if (eqn_no == 4) {
        /* ds equation */
        libxsmm_blasint my_eqn4 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(
            my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD);
        meqn_push_binary_op(my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_arg(my_eqn4, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn4, S3, S1, tmp_ld, 6, 0, bg_dt);
        meqn_push_arg(my_eqn4, S3, S1, ld, 0, 0, in_dt);
        func = meqn_dispatch(1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn4);
      } else if (eqn_no == 5) {
        /* din equation */
        libxsmm_blasint my_eqn5 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_binary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn5, S3, S1, tmp_ld, 6, 0, bg_dt);
        meqn_push_arg(my_eqn5, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn5, S3, S1, ld, 3, 0, in_dt);
        meqn_push_ternary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn5, S3, S1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn5, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn5, 1, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32);
        func = meqn_dispatch(S3, S1, &ld, in_dt, my_eqn5);
      } else {
        TPP_ASSERT(false, "LayerNormBwdTPP: invalid eqn. number %d\n", eqn_no);
      }
      return (void*)func;
    }

   private:
    int S1, S2, S3, eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  Eqn dgamma_func, dbeta_func, db_func, ds_func, din_func;
};

template <typename T>
class GroupNormFwdTPP {
 public:
  GroupNormFwdTPP() {}
  GroupNormFwdTPP(int S1, int S2, int S3, float eps)
      : S1(S1),
        S2(S2),
        S3(S3),
        eps(eps),
        reduce_cols_kernel(
            S1,
            S3,
            S2 * S3,
            S3,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD),
        reduce_rows_kernel(
            1,
            S3,
            S3,
            1,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        eqn(S1, S2, S3) {}
  void operator()(T* inp, T* gamma, T* beta, float* mean, float* var, T* out) {
    LIBXSMM_ALIGNED(float tmp[2 * S3], 64);
    const float c = 1.0 / ((float)S1 * S3);
    float m, v, s, b;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[5];
    eqn_param.inputs = arg_array;
    arg_array[1].primary = &s;
    arg_array[2].primary = &b;
    arg_array[3].primary = (void*)gamma;
    arg_array[4].primary = (void*)beta;
    for (int s2 = 0; s2 < S2; s2++) {
      reduce_cols_kernel((void*)&inp[s2 * S3], (void*)tmp);
      reduce_rows_kernel((void*)tmp, (void*)&m);
      reduce_rows_kernel((void*)&tmp[S3], (void*)&v);
      m = m * c;
      v = v * c;
      v = LIBXSMM_MAX(v - m * m, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      mean[s2] = m;
      var[s2] = v;
      s = v;
      b = -1.0 * v * m;
      arg_array[0].primary = (void*)&inp[s2 * S3];
      eqn_param.output.primary = (void*)&out[s2 * S3];
      eqn(&eqn_param);
    }
  }
  void ref(T* pinp, T* gamma, T* beta, float* mean, float* var, T* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, out, pout, S2, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float m = 0;
      float v = 0;
      float c = 1.0 / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          m += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
          v += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
        }
      }
      m = m * c;
      v = v * c;
      v = LIBXSMM_MAX(v - m * m, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      mean[s2] = m;
      var[s2] = v;
      float s = v;
      float b = -1.0 * v * m;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) =
              (LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) * s + b) *
                  gamma[s1] +
              beta[s1];
        }
      }
    }
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "group_norm_fwd_eqn_t%d_S1%d_S2%d_S3%d",
          XsmmDtype<T>(),
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = 1;
      libxsmm_blasint tmp_ld2 = S3;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
      meqn_push_ternary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1 |
              LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2 |
              LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
      meqn_push_ternary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
              LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
              LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, in_dt);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, 1, S1, 1, 3, 0, in_dt);
      meqn_push_arg(my_eqn0, 1, S1, 1, 4, 0, in_dt);
      debug_print_eqn_tree(my_eqn0); // printf
      return (void*)meqn_dispatch(S3, S1, &ld, out_dt, my_eqn0);
    }

   private:
    int S1, S2, S3;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  float eps;
  UnaryTPP reduce_cols_kernel;
  UnaryTPP reduce_rows_kernel;
  Eqn eqn;
};

template <typename T>
class GroupNormBwdTPP {
 public:
  GroupNormBwdTPP() {}
  GroupNormBwdTPP(int S1, int S2, int S3)
      : S1(S1),
        S2(S2),
        S3(S3),
        dgamma_func(S1, S2, S3, 1),
        dbeta_func(S1, S2, S3, 2),
        db_func(S1, S2, S3, 3),
        ds_func(S1, S2, S3, 4),
        din_func(S1, S2, S3, 5) {}
  void operator()(
      T* dout,
      T* inp,
      float* mean,
      float* var,
      T* gamma,
      T* din,
      float* dgamma,
      float* dbeta) {
    float a, b, c, db, ds;
    const float scale = 1.0f / ((float)S1 * S3);
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[8];
    eqn_param.inputs = arg_array;

    arg_array[1].primary = &a;
    arg_array[2].primary = &b;
    arg_array[4].primary = (void*)dgamma;
    arg_array[5].primary = (void*)dbeta;
    arg_array[6].primary = (void*)gamma;
    arg_array[7].primary = &c;

    for (int s2 = 0; s2 < S2; s2++) {
      a = var[s2];
      b = -a * mean[s2];
      arg_array[0].primary = (void*)&inp[s2 * S3];
      arg_array[3].primary = (void*)&dout[s2 * S3];

      eqn_param.output.primary = &ds;
      ds_func(&eqn_param);

      eqn_param.output.primary = &db;
      db_func(&eqn_param);

      eqn_param.output.primary = (void*)dgamma;
      dgamma_func(&eqn_param);

      eqn_param.output.primary = (void*)dbeta;
      dbeta_func(&eqn_param);

      b = (db * mean[s2] - ds) * a * a * a * scale;
      c = -b * mean[s2] - db * a * scale;

      eqn_param.output.primary = (void*)&din[s2 * S3];
      din_func(&eqn_param);
    }
  }
  void ref(
      T* pdout,
      T* pinp,
      float* mean,
      float* var,
      T* gamma,
      T* pdin,
      float* dgamma,
      float* dbeta) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, din, pdin, S2, S3);
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, dout, pdout, S2, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float a = var[s2], c;
      float b = -a * mean[s2];
      float ds = 0.0f;
      float db = 0.0f;
      float scale = 1.0f / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          dgamma[s1] +=
              (a * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + b) *
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
          dbeta[s1] += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
          ds += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * gamma[s1] *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
          db += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * gamma[s1];
        }
      }
      b = (db * mean[s2] - ds) * a * a * a * scale;
      c = -b * mean[s2] - db * a * scale;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3) =
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * a * gamma[s1] +
              b * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + c;
        }
      }
    }
  }

  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3, int eqn_no)
        : S1(S1), S2(S2), S3(S3), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "group_norm_bwd_eqn%d_t%d_S1%d_S2%d_S3%d",
          eqn_no,
          XsmmDtype<T>(),
          S1,
          S2,
          S3);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      // auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint tmp_ld2 = 1;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_matrix_eqn_function func = NULL;
      if (eqn_no == 1) {
        /* dgamma function  */
        libxsmm_blasint my_eqn1 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_unary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_binary_op(my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, S3, S1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn1, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn1, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn1, S3, S1, tmp_ld, 4, 0, LIBXSMM_DATATYPE_F32);
        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn1);
      } else if (eqn_no == 2) {
        /* dbeta function  */
        libxsmm_blasint my_eqn2 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_unary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_arg(my_eqn2, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn2, S3, S1, tmp_ld, 5, 0, LIBXSMM_DATATYPE_F32);
        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2);
      } else if (eqn_no == 3) {
        /* db equation */
        libxsmm_blasint my_eqn3 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1);
        meqn_push_arg(my_eqn3, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn3, 1, S1, 1, 6, 0, in_dt);
        func = meqn_dispatch(1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn3);
      } else if (eqn_no == 4) {
        /* ds equation */
        libxsmm_blasint my_eqn4 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(
            my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD);
        meqn_push_binary_op(
            my_eqn4,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1);
        meqn_push_arg(my_eqn4, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn4, 1, S1, 1, 6, 0, in_dt);
        meqn_push_arg(my_eqn4, S3, S1, ld, 0, 0, in_dt);
        func = meqn_dispatch(1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn4);
      } else if (eqn_no == 5) {
        /* din equation */
        libxsmm_blasint my_eqn5 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_binary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0 |
                LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn5, 1, S1, 1, 6, 0, in_dt);
        meqn_push_arg(my_eqn5, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn5, S3, S1, ld, 3, 0, in_dt);
        meqn_push_ternary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn5, S3, S1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn5, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn5, 1, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32);
        func = meqn_dispatch(S3, S1, &ld, in_dt, my_eqn5);
      } else {
        TPP_ASSERT(false, "GroupNormBwdTPP: invalid eqn. number %d\n", eqn_no);
      }
      return (void*)func;
    }

   private:
    int S1, S2, S3, eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  Eqn dgamma_func, dbeta_func, db_func, ds_func, din_func;
};

class SplitSGDTPP : public BaseTPP {
 public:
  SplitSGDTPP() {}
  SplitSGDTPP(int N) : N(N) {
    kernel = (libxsmm_matrix_eqn_function)get_kernel();
    initialized = true;
  }
  void operator()(bfloat16* hi, bfloat16* lo, bfloat16* grad, float lr) {
    if (!initialized)
      return;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[4];
    arg_array[0].primary = (void*)lo;
    arg_array[1].primary = (void*)hi;
    arg_array[2].primary = (void*)&lr;
    arg_array[3].primary = (void*)grad;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)lo;
    auto offset = (long long)((char*)hi - (char*)lo);
    eqn_param.output.secondary = (void*)offset;

    kernel(&eqn_param);
  }
  void ref(bfloat16* hi, bfloat16* lo, bfloat16* grad, float lr) {
#ifndef __AVX512F__
    auto dwt = (libxsmm_bfloat16*)grad;
    auto out_hi = (libxsmm_bfloat16*)hi;
    auto out_lo = (libxsmm_bfloat16*)lo;
    for (int i = 0; i < N; i++) {
      union libxsmm_bfloat16_f32 bf16_hp;
      union libxsmm_bfloat16_f32 bf16_wt;
      bf16_wt.i[0] = 0;
      bf16_wt.i[1] = dwt[i];
      bf16_hp.i[0] = out_lo[i];
      bf16_hp.i[1] = out_hi[i];
      bf16_hp.f = bf16_wt.f * lr + bf16_hp.f;
      out_lo[i] = bf16_hp.i[0];
      out_hi[i] = bf16_hp.i[1];
    }
#else
    long sz = N;
    auto vlr = _mm512_set1_ps(lr);
    long i;
    for (i = 0; i < ALIGNDOWN(sz, 16); i += 16) {
      auto grad_i = _mm512_loadu_ps_auto(&grad[i]);
      auto data_i = _mm512_split_loadu_ps(&hi[i], &lo[i]);
      data_i = _mm512_add_ps(data_i, _mm512_mul_ps(grad_i, vlr));
      _mm512_split_storeu_ps(&hi[i], &lo[i], data_i);
    }
    if (i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto grad_i = _mm512_maskz_loadu_ps_auto(mask, &grad[i]);
      auto data_i = _mm512_maskz_split_loadu_ps(mask, &hi[i], &lo[i]);
      data_i = _mm512_add_ps(data_i, _mm512_mul_ps(grad_i, vlr));
      _mm512_mask_split_storeu_ps(&hi[i], &lo[i], mask, data_i);
    }
#endif
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(hash, 200, "split_sgd_eqn_i%d", N);
    return std::string(hash);
  }
  void* build_kernel() override {
    libxsmm_blasint ld = N;
    libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
    meqn_push_unary_op(my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS);
    meqn_push_ternary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
        LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
    /* This is the "gradient" weights   */
    meqn_push_arg(my_eqn0, N, 1, ld, 3, 0, LIBXSMM_DATATYPE_BF16);
    /* This is the scalar learning rate */
    meqn_push_arg(my_eqn0, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_PACK);
    /* This is the tensor with lo bits  */
    meqn_push_arg(my_eqn0, N, 1, ld, 0, 0, LIBXSMM_DATATYPE_I16);
    /* This is the tensor with hi bits  */
    meqn_push_arg(my_eqn0, N, 1, ld, 1, 0, LIBXSMM_DATATYPE_I16);
    debug_print_eqn_tree(my_eqn0);
    auto func0 = meqn_dispatch(N, 1, &ld, LIBXSMM_DATATYPE_I16, my_eqn0);
    return (void*)func0;
  }

 private:
  int N = 0;
  libxsmm_matrix_eqn_function kernel = NULL;
};

class SplitSGDExtTPP {
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(SplitSGDExtTPP* p, int eqn_no) : p(p), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "splitsgdext_eqn%d_n%d",
          eqn_no,
          p->N);
      return std::string(hash);
    }
    void* build_kernel() override {

      int N = p->N;
      libxsmm_blasint ld = p->N;
      libxsmm_matrix_eqn_function func;

      if (eqn_no == 0) { /* weight decay step */
        libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();   /* grad_f32 = (d * weight_decay + grad_bf16) with d in split format */

        meqn_push_ternary_op(my_eqn0, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32);

        meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_PACK);

        /* This is the tensor with lo bits  */
        meqn_push_arg(my_eqn0, N, 1, ld, 0, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_I16); /* lo */

        /* This is the tensor with hi bits  */
        meqn_push_arg(my_eqn0, N, 1, ld, 1, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_I16); /*  hi */

        /* This is the scalar learning rate */
        meqn_push_arg(my_eqn0, N, 1, ld, 2, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* weight_decay (broadcast) */

        /* This is the "gradient" weights   */
        meqn_push_arg(my_eqn0, N, 1, ld, 3, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_BF16); /* grad */

        debug_print_eqn_tree(my_eqn0);
        func = meqn_dispatch( N, 1, &ld, LIBXSMM_DATATYPE_F32, my_eqn0); /* output is grad_f32 */
        if ( func == NULL) {
          fprintf( stderr, "JIT for TPP splitsgd weight_decay equation (eqn0) failed. Bailing...!\n");
          exit(-1);
        }
      } else if (eqn_no == 1) { /* momentum update step */
        libxsmm_blasint my_eqn1 = libxsmm_matrix_eqn_create();   /* m =  (1 - dampening) * grad + (momentum * m), all in fp32 */

        meqn_push_ternary_op(my_eqn1, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);

        meqn_push_arg(my_eqn1, N, 1, ld, 0, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* grad */

        meqn_push_arg(my_eqn1, N, 1, ld, 1, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* (1 - dampening) (broadcast) */

        meqn_push_binary_op(my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0);

        meqn_push_arg(my_eqn1, N, 1, ld, 2, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* momentum (broadcast) */

        meqn_push_arg(my_eqn1, N, 1, ld, 3, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* m */

        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(N, 1, &ld, LIBXSMM_DATATYPE_F32, my_eqn1); /* output is m */
        if ( func == NULL) {
          fprintf( stderr, "JIT for TPP splitsgd momentum equation (eqn1) failed. Bailing...!\n");
          exit(-1);
        }
      } else if (eqn_no == 2) { /* lr update step */
        libxsmm_blasint my_eqn2 = libxsmm_matrix_eqn_create();   /* d = m * (-lr) + d, with d in split format and rest as fp32 */

        meqn_push_unary_op(my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_BF16);

        meqn_push_ternary_op(my_eqn2, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);

        meqn_push_arg(my_eqn2, N, 1, ld, 0, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* m */

        meqn_push_arg(my_eqn2, N, 1, ld, 1, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* (- lr) (broadcast) */

        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_PACK);

        meqn_push_arg(my_eqn2, N, 1, ld, 2, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_I16); /* d, lo */

        meqn_push_arg(my_eqn2, N, 1, ld, 3, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_I16); /* d, hi */

        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(N, 1, &ld, LIBXSMM_DATATYPE_I16, my_eqn2); /* output is d (split) */
        if ( func == NULL) {
          fprintf( stderr, "JIT for TPP splitsgd lr update equation (eqn2) failed. Bailing...!\n");
          exit(-1);
        }
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    SplitSGDExtTPP* p;
    int eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  }; /* end of Eqn class */

 private:
  int N = 0;
  Eqn eqn_decay, eqn_momentum, eqn_lr;
  UnaryTPP copy_kernel;
  friend class Eqn;

 public:
  SplitSGDExtTPP(int N)
      : N(N),
        eqn_decay(this, 0),
        eqn_momentum(this, 1),
        eqn_lr(this, 2),
        copy_kernel(
            1, N, N, N,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY)
        { /*initialized = true;*/ }

  void operator()(bfloat16* d_lo, bfloat16* d_hi, bfloat16 *g_bf16, float* m, float *g_f32, float weight_decay, float dampening, float momentum, float lr, int step) {
    //if (!initialized)
    //  return;

    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[8];

    float alpha, beta;

    //std::cout << "dbg: before decay g_bf16 = " << g_bf16[0] << std::endl;
    //std::cout << "dbg: before decay d_hi   = " << d_hi[0] << std::endl;
    //std::cout << "dbg: before decay alpha  = " << alpha  << std::endl;

    alpha = weight_decay;
    arg_array[0].primary = (void*)d_lo;
    arg_array[1].primary = (void*)d_hi;
    arg_array[2].primary = (void*)&alpha;
    arg_array[3].primary = (void*)g_bf16;

    eqn_param.inputs         = arg_array;
    eqn_param.output.primary = g_f32;

    eqn_decay(&eqn_param);

    //printf("dbg: after decay g_f32 = %6.6f \n", g_f32[0]);

    if (step == 0) {
      copy_kernel(g_f32, m);
      //printf("dbg: after copy (step == 0) m = %6.6f \n", m[0]);
    } else {

      //printf("dbg: before momentum m = %6.6f %6.6f\n", m[0], m[1]);

      // for the older version which attempted to have a single equation which includes ternary and scaling in-place but didn't work
      alpha = 1 - dampening;
      beta  = momentum;
      arg_array[0].primary = (void*)g_f32;
      arg_array[1].primary = (void*)&alpha;
      arg_array[2].primary = (void*)&beta;
      arg_array[3].primary = (void*)m;
      //printf("dbg: alpha = %3.3f beta = %3.3f m = %6.6f g_f32 = %6.6f \n", alpha, beta, m[0], g_f32[0]);

      eqn_param.inputs           = arg_array;
      eqn_param.output.primary   = (void*)m;

      eqn_momentum(&eqn_param);
    }
    //printf("dbg: after momentum m = %6.6f %6.6f\n", m[0], m[1]);

    alpha = - lr;
    arg_array[0].primary = (void*)m;
    arg_array[1].primary = (void*)&alpha;
    arg_array[2].primary = (void*)d_lo;
    arg_array[3].primary = (void*)d_hi;

    eqn_param.inputs           = arg_array;
    eqn_param.output.primary   = (void*)d_lo;
    long long offset           = (long long) ((char*)d_hi - (char*)d_lo);
    eqn_param.output.secondary = (void*)offset;

    eqn_lr(&eqn_param);

    //std::cout << "dbg: after lr d_lo  = " << d_lo[0] << std::endl;
    //std::cout << "dbg: after lr d_hi  = " << d_hi[0] << std::endl;
    //std::cout << "dbg: after lr g_bf16 = " << g_bf16[0] << std::endl;

  }
  void ref(bfloat16* d_lo, bfloat16* d_hi, bfloat16 *g_bf16, float* m, float *g_f32, float weight_decay, float dampening, float momentum, float lr, int step) {
    printf("ref() not implemented for SplitSGDExtTPP\n");
    exit(-1);
  }
};

template <typename T>
class SGDExtTPP {
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(SGDExtTPP* p, int eqn_no) : p(p), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "sgdext_eqn%d_n%d_t%d",
          eqn_no,
          p->N,
          XsmmDtype<T>());
      return std::string(hash);
    }
    void* build_kernel() override {

      int N = p->N;
      libxsmm_blasint ld = p->N;
      libxsmm_matrix_eqn_function func;

      libxsmm_datatype datatype_in   = XsmmDtype<T>();
      libxsmm_datatype datatype_out  = datatype_in;
      libxsmm_datatype datatype_comp = XsmmDtype<T>();

      if (eqn_no == 0) { /* weight decay step */
        libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();   /* grad = (d * (-weight_decay) + grad), tensors in T, scalars fp32 */

        /* FIXME: Does this need a REUSE flag? */
        meqn_push_ternary_op(my_eqn0, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1, datatype_comp);

        meqn_push_arg(my_eqn0, N, 1, ld, 0, 0 /* offs_in_pos */, datatype_in); /* d */

        meqn_push_arg(my_eqn0, N, 1, ld, 1, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* weight_decay (broadcast) */

        meqn_push_arg(my_eqn0, N, 1, ld, 2, 0 /* offs_in_pos */, datatype_in); /* grad */

        debug_print_eqn_tree(my_eqn0);
        func = meqn_dispatch(N, 1, &ld, datatype_out, my_eqn0); /* output is grad */
        if ( func == NULL) {
          fprintf( stderr, "JIT for TPP sgd weight_decay equation (eqn0) failed. Bailing...!\n");
          exit(-1);
        }
      } else if (eqn_no == 1) { /* momentum update step */
        libxsmm_blasint my_eqn1 = libxsmm_matrix_eqn_create();   /* m =  (1 - dampening) * grad + (momentum * m), grad in T, m in fp32, scalars fp32 */

        meqn_push_ternary_op(my_eqn1, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, datatype_comp);

        meqn_push_arg(my_eqn1, N, 1, ld, 0, 0 /* offs_in_pos */, datatype_in); /* grad */

        meqn_push_arg(my_eqn1, N, 1, ld, 1, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* (1 - dampening) (broadcast) */

        meqn_push_binary_op(my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0, datatype_comp);

        meqn_push_arg(my_eqn1, N, 1, ld, 2, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* momentum (broadcast) */

        meqn_push_arg(my_eqn1, N, 1, ld, 3, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* m */

        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(N, 1, &ld, LIBXSMM_DATATYPE_F32, my_eqn1); /* output is m */
        if ( func == NULL) {
          fprintf( stderr, "JIT for TPP sgd momentum equation (eqn1) failed. Bailing...!\n");
          exit(-1);
        }
      } else if (eqn_no == 2) { /* lr update step */
        libxsmm_blasint my_eqn2 = libxsmm_matrix_eqn_create();   /* d = m * (-lr) + d, d in T, m in fp32, scalars fp32 */

        meqn_push_ternary_op(my_eqn2, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, datatype_comp);

        meqn_push_arg(my_eqn2, N, 1, ld, 0, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* m */

        meqn_push_arg(my_eqn2, N, 1, ld, 1, 0 /* offs_in_pos */, LIBXSMM_DATATYPE_F32); /* (- lr) (broadcast) */

        meqn_push_arg(my_eqn2, N, 1, ld, 2, 0 /* offs_in_pos */, datatype_in); /* d */

        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(N, 1, &ld, datatype_out, my_eqn2); /* output is d */
        if ( func == NULL) {
          fprintf( stderr, "JIT for TPP sgd lr update equation (eqn2) failed. Bailing...!\n");
          exit(-1);
        }
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    SGDExtTPP* p;
    int eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  }; /* end of Eqn class */

 private:
  int N = 0;
  Eqn eqn_decay, eqn_momentum, eqn_lr;
  UnaryTPP copy_kernel;
  friend class Eqn;

 public:
  SGDExtTPP(int N)
      : N(N),
        eqn_decay(this, 0),
        eqn_momentum(this, 1),
        eqn_lr(this, 2),
        copy_kernel(
            1, N, N, N,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(), // LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY)
        { /*initialized = true;*/ }

  void operator()(T* d, T *g, float* m, float weight_decay, float dampening, float momentum, float lr, int step) {
    //if (!initialized)
    //  return;

    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[8];

    float alpha, beta;

    alpha = weight_decay;
    arg_array[0].primary = (void*)d;
    arg_array[1].primary = (void*)&alpha;
    arg_array[2].primary = (void*)g;

    eqn_param.inputs         = arg_array;
    eqn_param.output.primary = g;

    eqn_decay(&eqn_param);

    //printf("dbg: after decay g = %6.6f \n", g[0]);

    if (step == 0) {
      copy_kernel(g, m);
      //printf("dbg: after copy (step == 0) m = %6.6f \n", m[0]);
    } else {

      //printf("dbg: before momentum m = %6.6f %6.6f\n", m[0], m[1]);

      // for the older version which attempted to have a single equation which includes ternary and scaling in-place but didn't work
      alpha = 1 - dampening;
      beta  = momentum;
      arg_array[0].primary = (void*)g;
      arg_array[1].primary = (void*)&alpha;
      arg_array[2].primary = (void*)&beta;
      arg_array[3].primary = (void*)m;

      //printf("dbg: alpha = %3.3f beta = %3.3f m = %6.6f g = %6.6f \n", alpha, beta, m[0], g[0]);

      eqn_param.inputs           = arg_array;
      eqn_param.output.primary   = (void*)m;

      eqn_momentum(&eqn_param);
    }

    //printf("dbg: after momentum m = %6.6f %6.6f\n", m[0], m[1]);

    alpha = - lr;
    arg_array[0].primary = (void*)m;
    arg_array[1].primary = (void*)&alpha;
    arg_array[2].primary = (void*)d;

    eqn_param.inputs           = arg_array;
    eqn_param.output.primary   = (void*)d;

    eqn_lr(&eqn_param);
  }
  void ref(T* d, T *g, float* m, float weight_decay, float dampening, float momentum, float lr, int step) {
    printf("ref() not implemented for SGDExtTPP\n");
    exit(-1);
  }
};


#if 0
template <typename Tin, typename Tout, typename Tind>
class EmbBagFwdTPP {
 public:
  EmbBagFwdTPP() {}
  EmbBagFwdTPP(int E)
      : E(E),
        kernel(
            0,
            E,
            E,
            E,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            (libxsmm_meltw_unary_flags)(
                LIBXSMM_MELTW_FLAG_UNARY_REDUCE_XOR_ACC |
                (sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES
                                   : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES)),
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD) {}
  void operator()(Tout* output, Tin* weight, Tind* input, int N) {
    unsigned long long _N = N;
    kernel((void*)weight, (void*)input, (void*)&_N, (void*)output, NULL);
  }
  void ref(Tout* output, Tin* weight, Tind* input, int N) {
    for (long v = 0; v < E; v++)
      output[v] = 0;
    for (long s = 0; s < N; s++) {
      auto ind = input[s];
      for (long v = 0; v < E; v++)
        output[v] += weight[ind * E + v];
    }
  }

 private:
  int E;
  UnaryTPP kernel;
};
#endif

template <typename Tin, typename Tout>
class EmbBagBwdTPP {
 public:
  EmbBagBwdTPP() {}
  EmbBagBwdTPP(int E)
      : E(E),
        kernel(
            0,
            E,
            E,
            E,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tout>(),
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {}
  void operator()(Tin* in, Tout* out, uint64_t N) {
    kernel((void*)in, NULL, NULL, (void*)&N, NULL, NULL, (void*)out, NULL);
  }
  void ref(Tin* in, Tout* out, uint64_t N) {
    for (uint64_t i = 0; i < N; i++) {
      for (int v = 0; v < E; v++) {
        out[i * E + v] = in[v];
      }
    }
  }

 private:
  int E;
  UnaryTPP kernel;
};

template <typename T>
class FusedAdamWTPP {
 public:
  FusedAdamWTPP() {}
  FusedAdamWTPP(int N, float beta1, float beta2, float weight_decay, float eps)
      : N(N),
        beta1(beta1),
        beta2(beta2),
        weight_decay(weight_decay),
        eps(eps),
        eqn0(this, 0),
        eqn1(this, 1),
        eqn2(this, 2) {}
  void operator()(
      T* data,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      float step_size,
      float lr) {
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
    float lrwd_1 = 1.0f - lr * weight_decay;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[6];
    arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta1_1;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&beta1;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)exp_avg;
    eqn0(&eqn_param);

    // arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta2_1;
    arg_array[2].primary = (void*)exp_avg_sq;
    arg_array[3].primary = (void*)&beta2;
    eqn_param.output.primary = (void*)exp_avg_sq;
    eqn1(&eqn_param);

    arg_array[0].primary = (void*)exp_avg_sq;
    arg_array[1].primary = (void*)&eps;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&step_size;
    arg_array[4].primary = (void*)data;
    arg_array[5].primary = (void*)&lrwd_1;
    eqn_param.output.primary = (void*)data;
    eqn2(&eqn_param);
  }

  void ref(
      T* data,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      float step_size,
      float lr) {
    long sz = N;
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
#ifndef __AVX512F__
    for (long i = 0; i < sz; i++) {
      auto avg_i = exp_avg[i];
      auto avg_sq_i = exp_avg_sq[i];
      auto grad_i = grad[i];
      auto data_i = data[i];
      avg_i = avg_i * beta1 + grad_i * beta1_1;
      avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
      auto denom = sqrtf(avg_sq_i) + eps;
      data_i = data_i - step_size * (avg_i / denom);
      if (weight_decay > 0.0)
        data_i = data_i - data_i * lr * weight_decay;
      exp_avg[i] = avg_i;
      exp_avg_sq[i] = avg_sq_i;
      data[i] = data_i;
    }
#else
    auto vbeta1 = _mm512_set1_ps(beta1);
    auto vbeta1_1 = _mm512_set1_ps(beta1_1);
    auto vbeta2 = _mm512_set1_ps(beta2);
    auto vbeta2_1 = _mm512_set1_ps(beta2_1);
    auto veps = _mm512_set1_ps(eps);
    auto vstep_size = _mm512_set1_ps(step_size);
    auto vweight_decay = _mm512_set1_ps(lr * weight_decay);
    long i;
    for (i = 0; i < ALIGNDOWN(sz, 16); i += 16) {
      auto avg_i = _mm512_loadu_ps(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps(&grad[i]);
      auto data_i = _mm512_loadu_ps(&data[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(
          data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      // if (weight_decay > 0.0)
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
      _mm512_storeu_ps(&exp_avg[i], avg_i);
      _mm512_storeu_ps(&exp_avg_sq[i], avg_sq_i);
      _mm512_storeu_ps(&data[i], data_i);
    }
    if (i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps(mask, &grad[i]);
      auto data_i = _mm512_maskz_loadu_ps(mask, &data[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(
          data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      // if (weight_decay > 0.0)
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
      _mm512_mask_storeu_ps(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps(&exp_avg_sq[i], mask, avg_sq_i);
      _mm512_mask_storeu_ps(&data[i], mask, data_i);
    }
#endif
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(FusedAdamWTPP* p, int eqn_no) : p(p), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "fused_adamw_eqn%d_t%d_n%d_wd%d",
          eqn_no,
          XsmmDtype<T>(),
          p->N,
          (p->weight_decay == 0.0 ? 0 : 1));
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      libxsmm_blasint ld = p->N;
      auto N = p->N;
      int use_wd = p->weight_decay == 0.0 ? 0 : 1;
      libxsmm_matrix_eqn_function func;
      if (eqn_no == 0) {
        // Equation for exp_avg
        auto my_eqn0 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn0, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta1
        meqn_push_binary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn0, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta1_1
        debug_print_eqn_tree(my_eqn0);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn0);
      } else if (eqn_no == 1) {
        // Equation for exp_avg_sq
        auto my_eqn1 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, N, 1, ld, 2, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta2
        meqn_push_binary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_X2);
        meqn_push_arg(my_eqn1, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta2_1
        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn1);
      } else if (eqn_no == 2) {
        // Equation for data_i (with decay)
        auto my_eqn2 = libxsmm_matrix_eqn_create();
        if (use_wd == 1) {
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        }
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_SUB);
        meqn_push_arg(my_eqn2, N, 1, ld, 4, 0, in_dt); // data_i
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_DIV);
        meqn_push_arg(my_eqn2, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_SQRT);
        meqn_push_arg(my_eqn2, N, 1, ld, 0, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn2, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // eps
        meqn_push_arg(
            my_eqn2, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // step_size
        if (use_wd == 1) {
          // this scalar is (1-lr*weight_decay)
          meqn_push_arg(my_eqn2, 1, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32);
        }
        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn2);
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    FusedAdamWTPP* p;
    int eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int N = 0;
  float beta1, beta2, weight_decay, eps;
  Eqn eqn0, eqn1, eqn2;
  friend class Eqn;
};

class FusedSplitAdamWTPP {
 public:
  typedef bfloat16 T;
  FusedSplitAdamWTPP() {}
  FusedSplitAdamWTPP(
      int N,
      float beta1,
      float beta2,
      float weight_decay,
      float eps)
      : N(N),
        beta1(beta1),
        beta2(beta2),
        weight_decay(weight_decay),
        eps(eps),
        eqn0(this, 0),
        eqn1(this, 1),
        eqn2(this, 2) {}
  void operator()(
      T* hi,
      T* lo,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      float step_size,
      float lr) {
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
    float lrwd_1 = 1.0f - lr * weight_decay;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[7];
    arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta1_1;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&beta1;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)exp_avg;
    eqn0(&eqn_param);

    // arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta2_1;
    arg_array[2].primary = (void*)exp_avg_sq;
    arg_array[3].primary = (void*)&beta2;
    eqn_param.output.primary = (void*)exp_avg_sq;
    eqn1(&eqn_param);

    arg_array[0].primary = (void*)exp_avg_sq;
    arg_array[1].primary = (void*)&eps;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&step_size;
    arg_array[4].primary = (void*)lo;
    arg_array[5].primary = (void*)hi;
    arg_array[6].primary = (void*)&lrwd_1;
    eqn_param.output.primary = (void*)lo;
    auto offset = (long long)((char*)hi - (char*)lo);
    eqn_param.output.secondary = (void*)offset;
    eqn2(&eqn_param);
  }

  void ref(
      T* hi,
      T* lo,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      float step_size,
      float lr) {
    long sz = N;
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
#ifndef __AVX512F__
    for (long i = 0; i < sz; i++) {
      union libxsmm_bfloat16_f32 data_hp;
      float avg_i = exp_avg[i];
      float avg_sq_i = exp_avg_sq[i];
      float grad_i = grad[i];
      data_hp.i[0] = lo[i];
      data_hp.i[1] = hi[i];
      float data_i = data_hp.f;

      avg_i = avg_i * beta1 + grad_i * beta1_1;
      avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
      auto denom = sqrtf(avg_sq_i) + eps;
      data_i = data_i - step_size * (avg_i / denom);
      if (weight_decay > 0.0)
        data_i = data_i - data_i * lr * weight_decay;
      exp_avg[i] = avg_i;
      exp_avg_sq[i] = avg_sq_i;
      data_hp.f = data_i;
      lo[i] = data_hp.i[0];
      hi[i] = data_hp.i[1];
    }
#else
    auto vbeta1 = _mm512_set1_ps(beta1);
    auto vbeta1_1 = _mm512_set1_ps(beta1_1);
    auto vbeta2 = _mm512_set1_ps(beta2);
    auto vbeta2_1 = _mm512_set1_ps(beta2_1);
    auto veps = _mm512_set1_ps(eps);
    auto vstep_size = _mm512_set1_ps(step_size);
    auto vweight_decay = _mm512_set1_ps(lr * weight_decay);
    long i;
    for (i = 0; i < ALIGNDOWN(sz, 16); i += 16) {
      auto avg_i = _mm512_loadu_ps(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps(&grad[i]);
      auto data_i = _mm512_split_loadu_ps(&hi[i], &lo[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(
          data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      if (weight_decay > 0.0)
        data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
      _mm512_storeu_ps(&exp_avg[i], avg_i);
      _mm512_storeu_ps(&exp_avg_sq[i], avg_sq_i);
      _mm512_split_storeu_ps(&hi[i], &lo[i], data_i);
    }
    if (i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps(mask, &grad[i]);
      auto data_i = _mm512_maskz_split_loadu_ps(mask, &hi[i], &lo[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(
          data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      if (weight_decay > 0.0)
        data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
      _mm512_mask_storeu_ps(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps(&exp_avg_sq[i], mask, avg_sq_i);
      _mm512_mask_split_storeu_ps(&hi[i], &lo[i], mask, data_i);
    }
#endif
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(FusedSplitAdamWTPP* p, int eqn_no) : p(p), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "fused_split_adamw_eqn%d_t%d_n%d_wd%d",
          eqn_no,
          XsmmDtype<T>(),
          p->N,
          (p->weight_decay == 0.0 ? 0 : 1));
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      libxsmm_blasint ld = p->N;
      auto N = p->N;
      int use_wd = p->weight_decay == 0.0 ? 0 : 1;
      libxsmm_matrix_eqn_function func;
      if (eqn_no == 0) {
        // Equation for exp_avg
        auto my_eqn0 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn0, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta1
        meqn_push_binary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn0, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta1_1
        debug_print_eqn_tree(my_eqn0);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn0);
      } else if (eqn_no == 1) {
        // Equation for exp_avg_sq
        auto my_eqn1 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, N, 1, ld, 2, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta2
        meqn_push_binary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_X2);
        meqn_push_arg(my_eqn1, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta2_1
        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn1);
      } else if (eqn_no == 2) {
        // Equation for data_i (with decay)
        auto my_eqn2 = libxsmm_matrix_eqn_create();
        meqn_push_unary_op(my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS);
        if (use_wd == 1) {
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        }
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_SUB);
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_PACK);
        meqn_push_arg(
            my_eqn2, N, 1, ld, 4, 0, LIBXSMM_DATATYPE_I16); // data_i lo
        meqn_push_arg(
            my_eqn2, N, 1, ld, 5, 0, LIBXSMM_DATATYPE_I16); // data_i hi

        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_DIV);
        meqn_push_arg(my_eqn2, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_SQRT);
        meqn_push_arg(my_eqn2, N, 1, ld, 0, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn2, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // eps
        meqn_push_arg(
            my_eqn2, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // step_size
        if (use_wd == 1) {
          // this scalar is (1-lr*weight_decay)
          meqn_push_arg(my_eqn2, 1, 1, 1, 6, 0, LIBXSMM_DATATYPE_F32);
        }
        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(N, 1, &ld, LIBXSMM_DATATYPE_I16, my_eqn2);
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    FusedSplitAdamWTPP* p;
    int eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int N = 0;
  float beta1, beta2, weight_decay, eps;
  Eqn eqn0, eqn1, eqn2;
  friend class Eqn;
};

template <typename T>
class FusedAdamStepTPP {
 public:
  FusedAdamStepTPP() {}
  FusedAdamStepTPP(
      int N,
      float beta1,
      float beta2,
      float eps,
      bool use_weight_decay,
      bool use_bias_correction)
      : N(N),
        beta1(beta1),
        beta2(beta2),
        eps(eps),
        use_weight_decay(use_weight_decay),
        use_bias_correction(use_bias_correction),
        eqn0(this, 0),
        eqn1(this, 1),
        eqn2(this, 2) {}
  void operator()(
      T* data,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      T* adam_step,
      float weight_decay = 0.0,
      float exp_avg_scale = 1.0,
      float exp_avg_sq_scale = 1.0) {
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[7];
    arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta1_1;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&beta1;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)exp_avg;
    eqn0(&eqn_param);

    // arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta2_1;
    arg_array[2].primary = (void*)exp_avg_sq;
    arg_array[3].primary = (void*)&beta2;
    eqn_param.output.primary = (void*)exp_avg_sq;
    eqn1(&eqn_param);

    arg_array[0].primary = (void*)exp_avg_sq;
    arg_array[1].primary = (void*)&eps;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)data;
    arg_array[4].primary = (void*)&weight_decay;
    arg_array[5].primary = (void*)&exp_avg_scale;
    arg_array[6].primary = (void*)&exp_avg_sq_scale;
    eqn_param.output.primary = (void*)adam_step;
    eqn2(&eqn_param);
  }

  void ref(
      T* data,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      T* adam_step,
      float weight_decay = 0.0,
      float exp_avg_scale = 1.0,
      float exp_avg_sq_scale = 1.0) {
    long sz = N;
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
#ifndef __AVX512F__
    for (long i = 0; i < sz; i++) {
      float avg_i = exp_avg[i];
      float avg_sq_i = exp_avg_sq[i];
      float grad_i = grad[i];
      avg_i = avg_i * beta1 + grad_i * beta1_1;
      avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
      exp_avg[i] = avg_i;
      exp_avg_sq[i] = avg_sq_i;
      if (use_bias_correction) {
        avg_i = avg_i * exp_avg_scale;
        avg_sq_i = avg_sq_i * exp_avg_sq_scale;
      }
      float denom = sqrtf(avg_sq_i) + eps;
      float adam_step_i = avg_i / denom;
      if (use_weight_decay) {
        float data_i = data[i];
        adam_step_i += data_i * weight_decay;
      }
      adam_step[i] = adam_step_i;
    }
#else
    auto vbeta1 = _mm512_set1_ps(beta1);
    auto vbeta1_1 = _mm512_set1_ps(beta1_1);
    auto vbeta2 = _mm512_set1_ps(beta2);
    auto vbeta2_1 = _mm512_set1_ps(beta2_1);
    auto veps = _mm512_set1_ps(eps);
    // auto vstep_size = _mm512_set1_ps(step_size);
    auto vweight_decay = _mm512_set1_ps(weight_decay);
    auto vexp_avg_scale = _mm512_set1_ps(exp_avg_scale);
    auto vexp_avg_sq_scale = _mm512_set1_ps(exp_avg_sq_scale);
    long i;
    for (i = 0; i < ALIGNDOWN(sz, 16); i += 16) {
      auto avg_i = _mm512_loadu_ps_auto(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps_auto(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps_auto(&grad[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      _mm512_storeu_ps_auto(&exp_avg[i], avg_i);
      _mm512_storeu_ps_auto(&exp_avg_sq[i], avg_sq_i);
      if (use_bias_correction) {
        avg_i = _mm512_mul_ps(avg_i, vexp_avg_scale);
        avg_sq_i = _mm512_mul_ps(avg_sq_i, vexp_avg_sq_scale);
      }
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      auto adam_step_i = _mm512_div_ps(avg_i, denom);
      if (use_weight_decay) {
        auto data_i = _mm512_loadu_ps_auto(&data[i]);
        adam_step_i =
            _mm512_add_ps(adam_step_i, _mm512_mul_ps(data_i, vweight_decay));
      }
      _mm512_storeu_ps_auto(&adam_step[i], adam_step_i);
    }
    if (i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps_auto(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps_auto(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps_auto(mask, &grad[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      _mm512_mask_storeu_ps_auto(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps_auto(&exp_avg_sq[i], mask, avg_sq_i);
      if (use_bias_correction) {
        avg_i = _mm512_mul_ps(avg_i, vexp_avg_scale);
        avg_sq_i = _mm512_mul_ps(avg_sq_i, vexp_avg_sq_scale);
      }
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      auto adam_step_i = _mm512_div_ps(avg_i, denom);
      if (use_weight_decay) {
        auto data_i = _mm512_maskz_loadu_ps_auto(mask, &data[i]);
        adam_step_i =
            _mm512_add_ps(adam_step_i, _mm512_mul_ps(data_i, vweight_decay));
      }
      _mm512_mask_storeu_ps_auto(&adam_step[i], mask, adam_step_i);
    }
#endif
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(FusedAdamStepTPP* p, int eqn_no) : p(p), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "fused_adam_step_eqn%d_t%d_n%d_wd%d",
          eqn_no,
          XsmmDtype<T>(),
          p->N,
          p->use_weight_decay);
      return std::string(hash);
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      libxsmm_blasint ld = p->N;
      auto N = p->N;
      int use_wd = p->use_weight_decay;
      int use_bc = p->use_bias_correction;
      libxsmm_matrix_eqn_function func;
      if (eqn_no == 0) {
        // Equation for exp_avg
        auto my_eqn0 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn0, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta1
        meqn_push_binary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn0, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta1_1
        debug_print_eqn_tree(my_eqn0);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn0);
      } else if (eqn_no == 1) {
        // Equation for exp_avg_sq
        auto my_eqn1 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, N, 1, ld, 2, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta2
        meqn_push_binary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_X2);
        meqn_push_arg(my_eqn1, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta2_1
        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn1);
      } else if (eqn_no == 2) {
        // Equation for adam_step_i (with decay)
        auto my_eqn2 = libxsmm_matrix_eqn_create();
        if (use_wd == 1) {
          meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD);
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
          meqn_push_arg(my_eqn2, N, 1, ld, 3, 0, in_dt); // data_i
          // weight_decay
          meqn_push_arg(my_eqn2, 1, 1, 1, 4, 0, LIBXSMM_DATATYPE_F32);
        }
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_DIV);
        if (use_bc) {
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0);
          meqn_push_arg(
              my_eqn2, 1, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32); // avg_i_scale
        }
        meqn_push_arg(my_eqn2, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_SQRT);
        if (use_bc) {
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0);
          meqn_push_arg(
              my_eqn2, 1, 1, 1, 6, 0, LIBXSMM_DATATYPE_F32); // avg_sq_i_scale
        }
        meqn_push_arg(my_eqn2, N, 1, ld, 0, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn2, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // eps
        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn2);
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    FusedAdamStepTPP* p;
    int eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int N = 0;
  float beta1, beta2, eps;
  bool use_weight_decay, use_bias_correction;
  Eqn eqn0, eqn1, eqn2;
  friend class Eqn;
};

template <typename Tin, typename Tout>
class ReduceColsTPP {
 public:
  ReduceColsTPP() {}
  ReduceColsTPP(int rows, int cols) : ReduceColsTPP(rows, cols, cols, cols) {}
  ReduceColsTPP(int rows, int cols, int ldi, int ldo) : ReduceColsTPP(rows, cols, ldi, ldo, 1) {}
  ReduceColsTPP(int rows, int cols, int ldi, int ldo, int reduce_on_output)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        reduce_on_output(reduce_on_output),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS | (reduce_on_output == 0 ? LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC : 0),
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  /* FIXME: To be checked */
  void ref(Tin* in, Tout* out) {
    for (int r = 0; r < rows; r++) {
      if (reduce_on_output == 0) { /* similar to beta = 0.0 gemm */
        out[r] = 0;
        out[rows + r] = 0;
      }
      float acc1 = 0.0, acc2 = 0.0;
      for (int c = 0; c < cols; c++) {
        acc1 += upconvert_to_float(in[r * ldi + c]);
        acc2 += upconvert_to_float(in[r * ldi + c]) * upconvert_to_float(in[r * ldi + c]);
      }
      out[       r] += acc1;
      out[rows + r] += acc2;
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  int reduce_on_output = 0; /* 0: initialize output with zeros; 1: only do += (default in the past but changed now in LIBXSMM) */
  UnaryTPP kernel;
};

/* Computes mean and var (per channel) from the arrays with sum and sum of squares (per channel) */
template <typename Tin, typename Tout = Tin>
class MeanVarTPP {
 private:
  int   N;
  Tout  scale;

 public:
  MeanVarTPP(int N, Tout scale)
      : N(N),
        scale(scale) {}
  void operator()(Tin* sum_x, Tin* sumsq_x, Tout* mean, Tout *var) {
    ref(sum_x, sumsq_x, mean, var);
  }
  void ref(Tin* sum_x, Tin* sumsq_x, Tout* mean, Tout *var) {
    for(int i = 0; i < N; i++){
      mean[i] = (float)sum_x  [i] * (float)scale;                          /* mean ~ E[X] */
      var[i]  = (float)sumsq_x[i] * (float)scale - mean[i] * mean[i];      /* var         */
      //printf("i = %d sum_x = %f simsq_x = %f mean = %f var = %10.10f \n", i, sum_x[i], sumsq_x[i], mean[i], var[i]);
    }
  }
};

template <typename Tin, typename Tout = Tin>
class BatchNormStatCoeffsTPP {
 private:
  int   N;
  Tout  eps;

 public:
  BatchNormStatCoeffsTPP(int N, Tout eps)
      : N(N),
        eps(eps) {}
  void operator()(Tin* mean, Tin* var, Tout* s, Tout* b) {
    ref(mean, var, s, b);
  }
  void ref(Tin* mean, Tin* var, Tout* s, Tout* b) {
    for(int i = 0; i < N; i++){
      s[i] = 1.0f / ((float)sqrt((float)var[i] + (float)eps));                          /* s = 1/sqrt(var(X) + eps)     [bc] */
      b[i] = - 1.0f * (float)mean[i] * s[i];                                            /* b = -E[X]/sqrt(var(X) + eps) [bc] */
    }
  }
};

template <typename Tin, typename Tout = Tin>
class BatchNormABCCoeffsTPP {
 private:
  int   N;
  Tout  scale;
  Tout  eps;

 public:
  BatchNormABCCoeffsTPP(int N, Tout scale, Tout eps)
      : N(N),
        scale(scale),
        eps(eps) {}
  void operator()(Tin* gamma, Tin* dgamma, Tin* var, Tin* mean, Tin* dbeta, Tout* a, Tout* b, Tout* c) {
    ref(gamma, dgamma, var, mean, dbeta, a, b, c);
  }
  void ref(Tin* gamma, Tin* dgamma, Tin* var, Tin* mean, Tin* dbeta, Tout* a, Tout* b, Tout* c) {
    for(int i = 0; i < N; i++){
      a[i] = gamma[i] / ((float)sqrt(var[i] + (float)eps));
      b[i] = - a[i] * (float)scale * dgamma[i] / ((float)sqrt(var[i] + (float)eps));
      c[i] = - b[i] * mean[i] - a[i] * scale * dbeta[i];
    }
  }
};

template <typename Tin, typename Tout>
class BatchNormFwdScaleTPP : public BaseTPP {
 private:
  int m = 0;
  int n = 0;
  int ldi = 0;
  int ldo = 0;
  typedef enum libxsmm_dnn_bn_fuse {
    LIBXSMM_DNN_BN_FUSE_NONE = 0,
    LIBXSMM_DNN_BN_FUSE_RELU = 1,
    LIBXSMM_DNN_BN_FUSE_ELTWISE = 2,
    LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU = 3,
    LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK = 4,
    LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK = 5
    } libxsmm_dnn_bn_fuse;
  /* fuse_type_int: 0: nothing fused, 1: relu fused, 2: ewise fused, 3: relu and ewise fused, 4: relu with mask, 5: relu and ewise with mask  */
  libxsmm_dnn_bn_fuse set_fuse_type(bool relu, bool eltwise) {
    libxsmm_dnn_bn_fuse res = LIBXSMM_DNN_BN_FUSE_NONE;
      if (relu == true) {
        if (eltwise == true)
          res = LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK; /* 5 # elwise+relu+mask */
        else
          res = LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK;         /* 4 # relu+mask */
      } else {
        if (eltwise == true)
          res = LIBXSMM_DNN_BN_FUSE_ELTWISE;                /* 2 # eltwise+no mask */
        else
          res = LIBXSMM_DNN_BN_FUSE_NONE;                   /* 0 no fusion */
      }
      return res;
  }
  libxsmm_dnn_bn_fuse fuse_type      = LIBXSMM_DNN_BN_FUSE_NONE;
  libxsmm_matrix_eqn_function kernel = NULL;

 public:
  BatchNormFwdScaleTPP() {
    initialized = false;
  }
  BatchNormFwdScaleTPP(int M, int N, int ldi, int ldo, bool relu, bool eltwise)
  : m(M), n(N), ldi(ldi), ldo(ldo) {
//    printf("m n ldi ldo = %d %d %d %d\n", m, n, ldi, ldo);
    if (ldi != m || ldo != m) {
      //printf("Case ldi or ldo != m  is not implemented in BatchNormFwdScaleTPP \n");
      //exit(-1);
//      printf("m n ldi ldo = %d %d %d %d\n", m, n, ldi, ldo);
    }
    fuse_type = set_fuse_type(relu, eltwise);
    kernel = (libxsmm_matrix_eqn_function)get_kernel();
    initialized = true;
  }
  BatchNormFwdScaleTPP(int M, int N, bool relu, bool eltwise) : BatchNormFwdScaleTPP(M, N, M, M, relu, eltwise) {}

  void operator()(Tin* inp, float* s, float* b, float *gamma, float *beta, Tin *inp_add, Tout* out, unsigned char* relumask) {
    if (!initialized)
      return;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[6];

    arg_array[0].primary = (void*)inp;
    arg_array[1].primary = (void*)s;
    arg_array[2].primary = (void*)b;
    arg_array[3].primary = (void*)gamma;
    arg_array[4].primary = (void*)beta;
    if (fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU ||  fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
      arg_array[5].primary = (void*)inp_add;
    }

    eqn_param.inputs         = arg_array;
    eqn_param.output.primary = out;
    if (fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
      eqn_param.output.secondary = ((fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                      (void*)relumask : NULL );
    }

    kernel(&eqn_param);
  }
  void ref(Tin* inp, float* s, float* b, float *gamma, float *beta, Tin *inp_add, Tout* out, unsigned char* relumask) {
    printf("ref() not implemented for BatchNormFwdScaleTPP\n");
    exit(-1);
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(hash, 200, "batchnorm_fwd_scale_ti%d_to%d_m%d_n%d_ldi%d_ldo%d_fuse%d", XsmmDtype<Tin>(), XsmmDtype<Tout>(), m, n, ldi, ldo, (int)fuse_type);
    return std::string(hash);
  }
  void* build_kernel() override {

    libxsmm_datatype datatype_in   = XsmmDtype<Tin>();
    libxsmm_datatype datatype_out  = XsmmDtype<Tout>();
    libxsmm_datatype datatype_comp = LIBXSMM_DATATYPE_F32;

    //libxsmm_blasint ldi     = ldi;
    //libxsmm_blasint ldo     = ldo;
    libxsmm_blasint tmp_ld  = 1;
    libxsmm_blasint tmp_ld2 = 1;
    libxsmm_blasint my_eqn10 = libxsmm_matrix_eqn_create();                          /* y = relu ( ( (s*x + b)*gamma + beta ) + inp_add) */

    if (fuse_type == 1 || fuse_type == 3 || fuse_type == 4 || fuse_type == 5) {
#ifdef __x86_64__
      meqn_push_unary_op(my_eqn10, LIBXSMM_MELTW_TYPE_UNARY_RELU,
                          ( (fuse_type == 4 || fuse_type == 5) ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT : LIBXSMM_MELTW_FLAG_UNARY_NONE),
                          datatype_out);

      if (datatype_out == LIBXSMM_DATATYPE_BF16)
        meqn_push_unary_op(my_eqn10, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_MELTW_FLAG_UNARY_NONE, datatype_out);
#else
#  warning "On GVT3 (ARM with neov1) bf16 relu produces incorrect relu masks so one has to do fp32 relu (which is less efficient)"
      meqn_push_unary_op(my_eqn10, LIBXSMM_MELTW_TYPE_UNARY_RELU,
                          ( (fuse_type == 4 || fuse_type == 5) ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT : LIBXSMM_MELTW_FLAG_UNARY_NONE),
                          datatype_comp);
#endif

    }

    if (fuse_type == 2 || fuse_type == 3 || fuse_type == 5) {
      meqn_push_binary_op(my_eqn10, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, datatype_comp);
    }

    meqn_push_ternary_op(my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
                          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT,
                          datatype_comp);

    meqn_push_arg(my_eqn10, m, 1, tmp_ld2, 3, 0 /* offs_in_pos */, datatype_comp); /* gamma = [bc] */

    meqn_push_ternary_op(my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
                          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT,
                          datatype_comp);

    meqn_push_arg(my_eqn10, m, 1, tmp_ld, 1, 0 /* offs_in_pos */, datatype_comp); /* s = [bc] */

    meqn_push_arg(my_eqn10, m, n, ldi, 0, 0 /* offs_in_pos */, datatype_in); /* x = [HW, bc] */

    meqn_push_arg(my_eqn10, m, 1, tmp_ld, 2, 0 /* offs_in_pos */, datatype_comp); /* b = [bc] */

    meqn_push_arg(my_eqn10, m, 1, tmp_ld2, 4, 0 /* offs_in_pos */, datatype_comp); /* beta = [bc] */

    if (fuse_type == 2 || fuse_type == 3 || fuse_type == 5) {
      meqn_push_arg(my_eqn10, m, n, ldi, 5, 0 /* offs_in_pos */, datatype_in); /* inp_add = [HW, bc] */
    }

    debug_print_eqn_tree(my_eqn10);
    auto func10 = meqn_dispatch(m, n, &ldo, datatype_out, my_eqn10); /* output is y = [HW, bc] */
    if ( func10 == NULL) {
      fprintf( stderr, "JIT for TPP fwd func10 (eqn10) failed. Bailing...!\n");
      exit(-1);
    }
    return (void*)func10;
  }
};


template <typename Tin, typename Tout>
class BatchNormBwdWTPP {
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(BatchNormBwdWTPP* p, int eqn_no) : p(p), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "batchnorm_bwd_w_eqn%d_ti%d_to%d_m%d_n%d",
          eqn_no,
          XsmmDtype<Tin>(),
          XsmmDtype<Tout>(),
          p->m,
          p->n);
      return std::string(hash);
    }
    void* build_kernel() override {
      libxsmm_datatype datatype_in   = XsmmDtype<Tin>();
      libxsmm_datatype datatype_out  = XsmmDtype<Tout>();
      libxsmm_datatype datatype_comp = LIBXSMM_DATATYPE_F32;

      libxsmm_blasint m       = p->m;
      libxsmm_blasint n       = p->n;
      libxsmm_blasint ld      = p->m;
      libxsmm_blasint tmp_ld2 = 1;
      libxsmm_matrix_eqn_function func;

      if (eqn_no == 0) {

        /* dgamma function  */
        auto my_eqn11 = libxsmm_matrix_eqn_create();                          /* dgamma = ((inp *a + b) * dout) + dgamma */

        meqn_push_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, datatype_comp);

        meqn_push_unary_op(my_eqn11, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, datatype_comp);

        meqn_push_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, datatype_comp);

        meqn_push_ternary_op(my_eqn11, LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
                              LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT,
                              datatype_comp);

        meqn_push_arg(my_eqn11, m, n, ld, 0, 0 /* offs_in_pos */, datatype_in); /* inp = [HW, bc] */

        meqn_push_arg(my_eqn11, m, 1, tmp_ld2, 1, 0 /* offs_in_pos */, datatype_comp); /* a = [bc] */

        meqn_push_arg(my_eqn11, m, 1, tmp_ld2, 2, 0 /* offs_in_pos */, datatype_comp); /* b = [bc] */

        meqn_push_arg(my_eqn11, m, n, ld, 3, 0 /* offs_in_pos */, datatype_out); /* dout = [HW, bc] */

        meqn_push_arg(my_eqn11, m, 1, tmp_ld2, 4, 0 /* offs_in_pos */, datatype_comp); /* dgamma = [bc] */

        debug_print_eqn_tree(my_eqn11);
        func = meqn_dispatch(m, 1, &tmp_ld2, datatype_comp, my_eqn11); /* output is dgamma [bc] */
        if ( func == NULL) {
          fprintf( stderr, "JIT for TPP bwd dgamma_func (eqn11) failed. Bailing...!\n");
          exit(-1);
        }
      } else if (eqn_no == 1) {
        /* dbeta function  */
        auto my_eqn12 = libxsmm_matrix_eqn_create();                         /* dbeta [bc] = dout [HW, bc] + dbeta [bc] */

        meqn_push_binary_op(my_eqn12, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, datatype_comp);

        meqn_push_unary_op(my_eqn12, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, datatype_comp);

        meqn_push_arg(my_eqn12, m, n, ld, 3, 0 /* offs_in_pos */, datatype_out); /* dout = [HW, bc] */

        meqn_push_arg(my_eqn12, m, 1, tmp_ld2, 5, 0 /* offs_in_pos */, datatype_comp); /* dbeta = [bc] */

        debug_print_eqn_tree(my_eqn12);
        func = meqn_dispatch(m, 1, &tmp_ld2, datatype_comp, my_eqn12); /* output is dbeta [bc] */
        if ( func == NULL) {
          fprintf( stderr, "JIT for TPP bwd dbeta_func (eqn12) failed. Bailing...!\n");
          exit(-1);
        }
      } else {
        TPP_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }

   private:
    BatchNormBwdWTPP* p;
    int eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  }; /* end of Eqn class */

 private:
  typedef enum libxsmm_dnn_bn_fuse {
    LIBXSMM_DNN_BN_FUSE_NONE = 0,
    LIBXSMM_DNN_BN_FUSE_RELU = 1,
    LIBXSMM_DNN_BN_FUSE_ELTWISE = 2,
    LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU = 3,
    LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK = 4,
    LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK = 5
    } libxsmm_dnn_bn_fuse;
  /* fuse_type_int: 0: nothing fused, 1: relu fused, 2: ewise fused, 3: relu and ewise fused, 4: relu with mask, 5: relu and ewise with mask  */
  libxsmm_dnn_bn_fuse set_fuse_type(bool relu, bool eltwise) {
    libxsmm_dnn_bn_fuse res = LIBXSMM_DNN_BN_FUSE_NONE;
      if (relu == true) {
        if (eltwise == true)
          res = LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK; /* 5 # elwise+relu+mask */
        else
          res = LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK;         /* 4 # relu+mask */
      } else {
        if (eltwise == true)
          res = LIBXSMM_DNN_BN_FUSE_ELTWISE;                /* 2 # eltwise+no mask */
        else
          res = LIBXSMM_DNN_BN_FUSE_NONE;                   /* 0 no fusion */
      }
      return res;
  }

 private:
  int m = 0;
  int n = 0;
  libxsmm_dnn_bn_fuse fuse_type      = LIBXSMM_DNN_BN_FUSE_NONE;
  Eqn eqn_dgamma, eqn_dbeta;
  UnaryTPP ewise_copy_kernel;
  UnaryTPP inv_relu_kernel;
  friend class Eqn;

 public:
  BatchNormBwdWTPP(int M, int N, bool relu, bool eltwise)
      : m(M),
        n(N),
        fuse_type(LIBXSMM_DNN_BN_FUSE_NONE),
        eqn_dgamma(this, 0),
        eqn_dbeta (this, 1),
        ewise_copy_kernel(
            n, m, m, m, // rows, cols, ldo, ldo due to UnaryTPP row-major-ness
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY),
        inv_relu_kernel(
            n, m, m, m, // rows, cols, ldo, ldo due to UnaryTPP row-major-ness
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT, /* always with bytemask here */
            LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) {
    fuse_type = set_fuse_type(relu, eltwise);
  }

  void operator()(Tin* inp, float *a, float *b, Tout *dout, float *dgamma_local, float *dbeta_local, float* gamma, Tin* din_add, unsigned char* relumask) {
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[8];

    arg_array[1].primary = (void*)a;
    arg_array[2].primary = (void*)b;
    arg_array[4].primary = (void*)dgamma_local;
    arg_array[5].primary = (void*)dbeta_local;
    arg_array[6].primary = (void*)gamma;

    if (fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE ||
      fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
      if (fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
        const float alpha = 0.0f;

        inv_relu_kernel((void*)dout, (fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ? (void*)relumask : NULL, NULL /*in.tertiary*/,
                        (void*)&alpha, NULL, NULL, /* op primary, secondary, tertiary */
                        (void*)dout, NULL);
      } /* ReLU/mask */
      if (fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
        ewise_copy_kernel(dout, din_add);
      } /* Eltwise */
    }

    arg_array[0].primary = (void*)inp;
    arg_array[3].primary = (void*)dout;

    eqn_param.inputs         = arg_array;

    eqn_param.output.primary = dgamma_local;
    eqn_dgamma(&eqn_param);

    eqn_param.output.primary = dbeta_local;
    eqn_dbeta(&eqn_param);

  }
  void ref(Tin* inp, float *a, float *b, Tout *dout, float *dgamma_local, float *dbeta_local, float* gamma, Tin* din_add, unsigned char* relumask) {
    printf("ref() not implemented for BatchNormBwdWTPP\n");
    exit(-1);
  }

};


template <typename Tin, typename Tout>
class BatchNormBwdDTPP : public BaseTPP {
 private:
  int m = 0;
  int n = 0;
  libxsmm_matrix_eqn_function kernel = NULL;

 public:
  BatchNormBwdDTPP(int M, int N) : m(M), n(N) {
    kernel = (libxsmm_matrix_eqn_function)get_kernel();
    initialized = true;
  }
  void operator()(Tin* inp, float* a, float* b, float *c, float *gamma, Tout* dout, Tout* din) {
    if (!initialized)
      return;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[8];

    arg_array[1].primary = (void*)a;
    arg_array[2].primary = (void*)b;
    arg_array[6].primary = (void*)gamma;
    arg_array[7].primary = (void*)c;

    arg_array[0].primary = (void*)inp;
    arg_array[3].primary = (void*)dout;

    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)din;
    kernel(&eqn_param);                                                                     /* din = dout * a + b * inp + c */

    kernel(&eqn_param);
  }
  void ref(Tin* inp, float* a, float* b, float *c, float *gamma, Tout* dout, Tout* din) {
    printf("ref() not implemented BatchNormBwdDTPP\n");
    exit(-1);
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(hash, 200, "batchnorm_bwd_d_ti%d_to%d_m%d_n%d", XsmmDtype<Tin>(), XsmmDtype<Tout>(), m, n);
    return std::string(hash);
  }
  void* build_kernel() override {

    libxsmm_datatype datatype_in   = XsmmDtype<Tin>();
    libxsmm_datatype datatype_out  = XsmmDtype<Tout>();
    libxsmm_datatype datatype_comp = LIBXSMM_DATATYPE_F32;

    libxsmm_blasint ld      = m;
    libxsmm_blasint tmp_ld2 = 1;
    /* din long equation */
    libxsmm_blasint my_eqn16 = libxsmm_matrix_eqn_create();                          /* din = a * dout + (b * inp + c) */

    meqn_push_ternary_op(my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
                          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT,
                          datatype_comp);

    meqn_push_arg(my_eqn16, m, 1, tmp_ld2, 1, 0 /* offs_in_pos */, datatype_comp); /* a = [bc] */

    meqn_push_arg(my_eqn16, m, n, ld, 3, 0 /* offs_in_pos */, datatype_out); /* dout = [HW, bc] */

    meqn_push_ternary_op(my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
                          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT,
                          datatype_comp);

    meqn_push_arg(my_eqn16, m, n, ld, 0, 0 /* offs_in_pos */, datatype_in); /* inp = [HW, bc] */

    meqn_push_arg(my_eqn16, m, 1, tmp_ld2, 2, 0 /* offs_in_pos */, datatype_comp); /* b = [bc] */

    meqn_push_arg(my_eqn16, m, 1, tmp_ld2, 7, 0 /* offs_in_pos */, datatype_comp); /* c = [bc] */

    debug_print_eqn_tree(my_eqn16);
    auto func = meqn_dispatch(m, n, &ld, datatype_out, my_eqn16); /* output is din [HW, bc] */
    if ( func == NULL) {
      fprintf( stderr, "JIT for TPP bwd din_func (eqn16) failed. Bailing...!\n");
      exit(-1);
    }
    return (void*)func;
  }
};

/* Unlike ReduceAddColTPP, accumulates to the prescribed datatype and not always to float */
template <typename Tin, typename Tout>
class ReduceAddColExtTPP {
 public:
  ReduceAddColExtTPP() {}
  ReduceAddColExtTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        reduce(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) {}
  void operator()(Tin* in, Tout* out) {
    reduce(in, out);
  }
  void ref(Tin* in, Tout* out) {
    for (int c = 0; c < cols; c++) {
      float acc = 0.0;
      for (int r = 0; r < rows; r++) {
        acc += upconvert_to_float(in[r * ldi + c]);
      }
      out[c] = acc;
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi, ldo;

  UnaryTPP reduce;
};

}; // namespace tpp

#endif // _XSMM_FUNCTORS_H_
