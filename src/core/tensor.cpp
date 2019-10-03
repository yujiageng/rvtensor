/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include <cassert>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <memory>
#include "include/core/tensor.hpp"

namespace RVTensor {

////////////////////// Tensor ////////////////////////////////
Tensor::sptr Tensor::create() {
  return std::make_shared<Tensor>();
}

Tensor::sptr Tensor::create(int n, int h, int w, int c, size_t elemsize) {
  return std::make_shared<Tensor>(n, h, w, c, elemsize);
}

Tensor::sptr Tensor::create(int n, int h, int w, int c,
                            void* data, size_t elemsize) {
  return std::make_shared<Tensor>(n, h, w, c, data, elemsize);
}

inline Tensor::Tensor() : data_ptr(nullptr), element_size(0), n_batch(0),
                          width(0), height(0), channel(0) {}

inline Tensor::Tensor(int n, int h, int w, int c, size_t elemsize)
  : data_ptr(nullptr), element_size(elemsize), n_batch(n), width(w),
                                                  height(h), channel(c) {}

inline Tensor::Tensor(int n, int h, int w, int c, void* data, size_t elemsize)
  : data_ptr(data), element_size(elemsize), n_batch(n), width(w),
                                                  height(h), channel(c) {}

inline Tensor::~Tensor() {
  element_size = 0;
  n_batch = 0;
  width = 0;
  height = 0;
  channel = 0;
}

inline void Tensor::reSize(int n, int h, int w, int c, size_t elemsize) {
    n_batch = n;
    height = h;
    width = w;
    channel = n;
    element_size = elemsize;
}

inline void Tensor::reLoadData(void* data) {
    if (data)
        data_ptr = data;
}

inline bool Tensor::empty() const {
  return data_ptr == nullptr || trueSize() == 0;
}

size_t Tensor::count() const {
  return n_batch * channel * height * width;
}

size_t Tensor::trueSize() const {
  return n_batch * channel * height * width * element_size;
}

/////////////////// FlashTensor /////////////////////////////
FlashTensor::sptr FlashTensor::create() {
  return std::make_shared<FlashTensor>();
}

FlashTensor::sptr FlashTensor::create(int n, int h, int w, int c,
                                      size_t elemsize) {
  return std::make_shared<FlashTensor>(n, h, w, c, elemsize);
}

FlashTensor::sptr FlashTensor::create(int n, int h, int w, int c, void* data,
                                      size_t elemsize) {
  return std::make_shared<FlashTensor>(n, h, w, c, data, elemsize);
}

inline FlashTensor::FlashTensor() : Tensor() {}

inline FlashTensor::FlashTensor(int n, int h, int w, int c, size_t elemsize)
  : Tensor(n, h, w, c, elemsize) {}

inline FlashTensor::FlashTensor(int n, int h, int w, int c,
                                void* data, size_t elemsize)
  : Tensor(n, h, w, c, data, elemsize) {}

inline FlashTensor::~FlashTensor() {
  data_ptr = nullptr;
}

void FlashTensor::bindModelData(void* data, size_t size) {
  if (size == trueSize() && data_ptr == nullptr)
    data_ptr = data;
  else
    throw std::runtime_error("FlashTensor duplicate copy of data_ptr!");
}

template <typename T>
  inline const T FlashTensor::grepElement(int n, int h, int w, int c) const {
    assert(sizeof(T) == element_size);
    return *(reinterpret_cast<const T*>(data_ptr) +
        (n * height * width * channel +
         h * width * channel + w * channel + c) * element_size);
}

template <typename T>
  inline FlashTensor::operator const T*() const {
    assert(sizeof(T) == element_size);
    return reinterpret_cast<const T*>(data_ptr);
  }

////////////////// RamTensor ////////////////////////////////
RamTensor::sptr RamTensor::create() {
  return std::make_shared<RamTensor>();
}

RamTensor::sptr RamTensor::create(int n, int h, int w, int c, size_t elemsize) {
  return std::make_shared<RamTensor>(n, h, w, c, elemsize);
}

RamTensor::sptr RamTensor::create(int n, int h, int w, int c,
    void* data, size_t elemsize) {
  return std::make_shared<RamTensor>(n, h, w, c, data, elemsize);
}

inline RamTensor::RamTensor() : Tensor() {}

inline RamTensor::RamTensor(int n, int h, int w, int c, size_t elemsize)
  : Tensor(n, h, w, c, elemsize), is_malloced(true) {

    if (trueSize() > 0) {
      size_t total_size = alignSize(trueSize() * elemsize, 4);
      data_ptr = tensorDataMalloc(total_size);
    }
}

inline RamTensor::RamTensor(int n, int h, int w, int c,
    void* data, size_t elemsize)
  : Tensor(n, h, w, c, data, elemsize), is_malloced(false) {}

inline RamTensor::~RamTensor() {
  if (is_malloced)
    tensorDataFree();
}

template <typename T>
void RamTensor::fill(T _v) {
  for (int i = 0; i < height * width; i++) {
    T* dst_ptr = reinterpret_cast<T*>(
                 reinterpret_cast<uint8_t*>(data_ptr) +
                 i * channel * element_size);
    for (int c = 0; c < channel; c++) {
      dst_ptr[c] = _v;
    }
  }
}

void RamTensor::fill(uint8_t v) {
  for (int i = 0; i < height * width; i++) {
    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(data_ptr) +
                       i * channel * element_size;
    memset(dst_ptr, v, channel);
  }
}

inline RamTensor::sptr RamTensor::clone() const {
  if (empty())
    return create();

  RamTensor::sptr ts = std::make_shared<RamTensor>(
      n_batch, height, width, channel, element_size);

  if (trueSize() > 0) {
    memcpy(ts->data_ptr, data_ptr, trueSize());
  }

  return ts;
}

void RamTensor::reConfigTensor(int n, int h, int w, int c, void* data,
                                                    size_t elemsize) {

    assert(is_malloced == false);
    reSize(n, h, w, c, elemsize);
    reLoadData(data);
}

void* RamTensor::tensorDataMalloc(size_t size) {
  unsigned char* udata = (unsigned char*)malloc(size +
                          sizeof(void*) + MALLOC_ALIGN);
  if (!udata)
    return 0;
  unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
  adata[-1] = udata;
  return adata;
}

void RamTensor::tensorDataFree() {
  if (data_ptr) {
    unsigned char* udata = ((unsigned char**)data_ptr)[-1];
    free(udata);
  }
}

void RamTensor::writeData(void* data, size_t size) {
  if (size == trueSize() && data_ptr != nullptr)
    memcpy(data_ptr, data, size);
  else
    throw std::runtime_error("RamTensor error in write data!");
}

template <typename T>
  inline T* RamTensor::grepBatchData(int n) {
    assert(sizeof(T) == element_size);
    return reinterpret_cast<T*>(data_ptr) +
      n * height * width * channel * element_size;
}

template <typename T>
  inline T RamTensor::grepElement(int n, int h, int w, int c) {
    assert(sizeof(T) == element_size);
    return *(reinterpret_cast<T*>(data_ptr) +
        (n * height * width * channel +
         h * width * channel + w * channel + c) * element_size);
}

template <typename T>
  inline RamTensor::operator T*() {
    assert(sizeof(T) == element_size);
    return reinterpret_cast<T*>(data_ptr);
  }

}  // namespace RVTensor
