#ifndef _PYTORCH_EXTENSION_WRAPPER_H_
#define _PYTORCH_EXTENSION_WRAPPER_H_

#ifdef __x86_64__
#include <immintrin.h>
#endif

#ifdef TORCH_API_INCLUDE_EXTENSION_H
#error "This file should not be included when building pytorch extension"
#endif

namespace at {

class BFlaoat16 {
 public:
  BFlaoat16(float f) {
    u_bf16 tmp;
    tmp.f = f;
    val = tmp.i[1];
  }
  operator float() {
    u_bf16 tmp;
    tmp.i[0] = 0;
    tmp.i[1] = val;
    return tmp.f;
  }

 protected:
  typedef union u_bf16 {
    float f;
    unsigned short i[2];
  };
  unsigned short val;
};

#ifdef __x86_64__
class Half {
 public:
  Half(float f) {
    val = _cvtss_sh(f);
  }
  operator float() {
    return _cvtsh_ss(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }

 protected:
  unsigned short val;
};
#elif defined(__aarch64__)
typedef __fp16 Half;
#else
#error "Unsupported architecture"
#endif

}; // namespace at

#endif // _PYTORCH_EXTENSION_WRAPPER_H_
