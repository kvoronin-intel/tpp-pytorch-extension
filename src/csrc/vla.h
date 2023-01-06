#ifndef _TPP_VLA_H_
#define _TPP_VLA_H_

#include <cstdint>
#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>
#endif

template <typename T, typename index_t = int64_t>
class VLAAccessorBase {
 public:
  typedef T* PtrType;

  VLAAccessorBase(PtrType data_, const index_t* strides_)
      : data_(data_), strides_(strides_) {}

 protected:
  PtrType data_;
  const index_t* strides_;
};

template <typename T, std::size_t N, typename index_t = int64_t>
class VLAAccessor : public VLAAccessorBase<T, index_t> {
 public:
  typedef T* PtrType;

  VLAAccessor(PtrType data_, const index_t* strides_)
      : VLAAccessorBase<T, index_t>(data_, strides_) {}

  VLAAccessor<T, N - 1, index_t> operator[](index_t i) {
    return VLAAccessor<T, N - 1, index_t>(
        this->data_ + this->strides_[0] * i, this->strides_ + 1);
  }

  const VLAAccessor<T, N - 1, index_t> operator[](index_t i) const {
    return VLAAccessor<T, N - 1, index_t>(
        this->data_ + this->strides_[0] * i, this->strides_ + 1);
  }
};

#if 1
template <typename T, typename index_t>
class VLAAccessor<T, 1, index_t> : public VLAAccessorBase<T, index_t> {
 public:
  typedef T* PtrType;

  VLAAccessor(PtrType data_, const index_t* strides_)
      : VLAAccessorBase<T, index_t>(data_, strides_) {}
  T* operator[](index_t i) {
    return this->data_ + i * this->strides_[0];
  }
  const T* operator[](index_t i) const {
    return this->data_ + i * this->strides_[0];
  }
};
#endif
template <typename T, typename index_t>
class VLAAccessor<T, 0, index_t> : public VLAAccessorBase<T, index_t> {
 public:
  typedef T* PtrType;

  VLAAccessor(PtrType data_, const index_t* strides_)
      : VLAAccessorBase<T, index_t>(data_, strides_) {}
  T& operator[](index_t i) {
    return this->data_[i];
  }
  const T& operator[](index_t i) const {
    return this->data_[i];
  }
  operator T*() {
    return this->data_;
  }
  operator const T*() const {
    return this->data_;
  }
};

template <typename T, std::size_t N, typename index_t = int64_t>
class VLAPtr {
 public:
  VLAPtr(T* data_, const index_t (&sizes)[N]) : data_(data_) {
    strides[N - 1] = sizes[N - 1];
    for (long i = N - 2; i >= 0; i--)
      strides[i] = strides[i + 1] * sizes[i];
  }
  VLAAccessor<T, N - 1, index_t> operator[](index_t i) {
    return VLAAccessor<T, N - 1, index_t>(data_ + i * strides[0], strides + 1);
  }
  operator bool() {
    return data_ != nullptr;
  }

 protected:
  index_t strides[N];
  T* data_;
};

#if 1
template <typename T>
class VLAPtr<T, 1, int64_t> {
 public:
  typedef int64_t index_t;
  VLAPtr(T* data_, const index_t (&sizes)[1]) : data_(data_) {
    strides[0] = sizes[0];
  }
  T* operator[](index_t i) {
    return data_ + i * strides[0];
  }
  operator bool() {
    return data_ != nullptr;
  }

 protected:
  index_t strides[1];
  T* data_;
};
#endif

template <typename T, std::size_t N, typename index_t = int64_t>
VLAPtr<T, N, index_t> GetVLAPtr(T* data_, const index_t (&list)[N]) {
  return VLAPtr<T, N, index_t>(data_, list);
}

#ifdef TORCH_API_INCLUDE_EXTENSION_H
template <typename T>
inline T* pt_get_data_ptr(at::Tensor t) {
  return t.data_ptr<T>();
}
template <typename T>
inline T* pt_get_data_ptr(at::Tensor t, unsigned long offset_in_T) {
  return t.data_ptr<T>() + offset_in_T;
}
template <typename T, typename Tbase>
inline T* pt_get_data_ptr(at::Tensor t, unsigned long offset_in_Tbase) {
  return reinterpret_cast<T*>(t.data_ptr<Tbase>() + offset_in_Tbase);
}
#ifndef PYTORCH_SUPPORTS_BFLOAT8
template <>
inline at::BFloat8* pt_get_data_ptr<at::BFloat8>(at::Tensor t) {
  return (at::BFloat8*)t.data_ptr<uint8_t>();
}
template <>
inline at::BFloat8* pt_get_data_ptr<at::BFloat8>(at::Tensor t, unsigned long offset_in_T) {
  return (at::BFloat8*)(t.data_ptr<uint8_t>() + offset_in_T);
}
#endif
typedef int64_t index_t;
template <typename T, std::size_t N> //, typename index_t = int64_t>
VLAPtr<T, N, index_t> GetVLAPtr(at::Tensor t, const index_t (&sizes)[N]) {
  return VLAPtr<T, N, index_t>(pt_get_data_ptr<T>(t), sizes);
}
template <typename T>
T* GetVLAPtr(at::Tensor t) {
  return pt_get_data_ptr<T>(t);
}
#endif

#endif // _TPP_VLA_H_
