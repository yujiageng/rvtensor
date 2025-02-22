/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_CORE_TENSOR_HPP_
#define INCLUDE_CORE_TENSOR_HPP_

#include <string.h>  //NOLINT
#include <memory>

namespace RVTensor {

/**
 * Aligns a buffer size to the specified number of bytes
 */
// static inline size_t alignSize(size_t sz, int n) { return (sz + n - 1) & -n; }

/**
 * the alignment of all the allocated buffers
 */
// #define MALLOC_ALIGN 16

/**Aligns a pointer to the specified number of bytes
 * @param ptr: Aligned pointer
 * @param n: Alignment size that must be a power of two
 */
// template <typename _Tp>
// static inline _Tp* alignPtr(_Tp* ptr, int n = sizeof(_Tp)) {
//   return reinterpret_cast<_Tp*>(((size_t)ptr + n - 1) & -n);
// }

/**
 * RVTensor data descriptor
 *
 *                 Tensor
 *                   +
 *                   |
 *       +-----------+-----------+
 *       |                       |
 *       v                       v
 *  FlashTensor               RamTensor
 *
 * FlashTensor describes model data which is already loaded in flash.
 * RamTensor describes dynamic memory data.
 *
 */
class Tensor {
 public:
  using sptr = std::shared_ptr<Tensor>;
  static sptr create();
  static sptr create(int n, int c, int h, int w, size_t elemsize = 4u);
  static sptr create(int n, int c, int h, int w, void* data,
                     size_t elemsize = 4u);

  /**
   * Constructor & Deconstructor
   */
  Tensor();
  Tensor(int n, int c, int h, int w, size_t elemsize = 4u);
  Tensor(int n, int c, int h, int w, void* data, size_t elemsize = 4u);
  ~Tensor();
  Tensor& operator=(const Tensor& m);

  void reSize(int n, int c, int h, int w, size_t elemsize = 4u);
  void reLoadData(void* data);

  /**
   *??tensor?????
   */
  bool empty() const;

  /**
   * Number of the elements in Tensor
   */
  size_t count() const;
  /**
   * True size of the elements in Tensor
   */
  size_t trueSize() const;

  /**
   * pointer to the data
   */
  void* data_ptr;

  /**
   * element size in bytes
   * 4 = float32/int32
   * 2 = float16
   * 1 = int8/uint8
   * 0 = empty
   */
  size_t element_size;

  /**
   * dimension of the Tensor
   */
  int n_batch;
  int width;
  int height;
  int channel;
};

class FlashTensor : public Tensor {
 public:
  using sptr = std::shared_ptr<FlashTensor>;
  static sptr create();
  static sptr create(int n, int c, int h, int w, size_t elemsize = 4u);
  static sptr create(int n, int c, int h, int w, void* data,
                     size_t elemsize = 4u);

  /**
   * Constructor & Deconstructor
   */
  FlashTensor();
  FlashTensor(int n, int c, int h, int w, size_t elemsize = 4u);
  FlashTensor(int n, int c, int h, int w, void* data, size_t elemsize = 4u);
  ~FlashTensor();
  FlashTensor& operator=(const FlashTensor& m);

  /**
   * bind model data to the Tensor
   */
  void bindModelData(void* data, size_t size);

  /**
   * get const data reference
   */
  template <typename T>
  const T* grepBatchData(int n) const;
  template <typename T>
  const T grepElement(int n, int c, int h, int w) const;
  template <typename T>
  operator const T*() const;
};

class RamTensor : public Tensor {
 public:
  using sptr = std::shared_ptr<RamTensor>;
  static sptr create();
  static sptr create(int n, int c, int h, int w, size_t elemsize = 4u);
  static sptr create(int n, int c, int h, int w, void* data,
                     size_t elemsize = 4u);

  /**
   * Constructor & Deconstructor
   */
  RamTensor();
  RamTensor(int n, int c, int h, int w, size_t elemsize);
  RamTensor(int n, int c, int h, int w, void* data, size_t elemsize);
  ~RamTensor();
  RamTensor& operator=(const RamTensor& m);

  /**
   * set tensor data with v
   */
  template <typename T>
  void fill(T v);
  void fill(uint8_t v);

  /**
   * deep copy of this Tensor
   */
  RamTensor::sptr clone() const;

  /**
   * reconfig tensor
   */
  void reConfigTensor(int n, int c, int h, int w, void* data,
                      size_t elemsize = 4u);

  /**
   *  write data to data_ptr
   */
  void writeData(void* data, size_t size);

  /**
   * get data reference
   */
  template <typename T>
  T* grepBatchData(int n);
  template <typename T>
  T grepElement(int n, int c, int h, int w);
  template <typename T>
  operator T*();

 private:
  /**
   *  memory malloc&free for data_ptr
   */
  void* tensorDataMalloc(size_t size);
  void tensorDataFree();
  /**
   * Is data_ptr malloced in Tensor/RamTensor
   */
  bool is_malloced;
};

}  // namespace RVTensor

#endif  // INCLUDE_CORE_TENSOR_HPP_
