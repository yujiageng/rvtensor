/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_CORE_EXECUTOR_HPP_
#define INCLUDE_CORE_EXECUTOR_HPP_

#include <string>
#include <memory>
#include <vector>
#include "include/core/tensor.hpp"

namespace rvos {

enum Direction_t{
  HOST_TO_AI = 0,
  AI_TO_HOST = 1,
};

typedef void (*callback_draw_box)(uint32_t x1, uint32_t y1, uint32_t x2,
    uint32_t y2, uint32_t classes, float prob);

/**
 * Executor describes the context of a individual inference task.
 */
class Executor {
 public:
    using sptr = std::shared_ptr<Executor>;
    static sptr create();
    static sptr create(std::string model_name, int input_height,
        int input_width, int thread_num);

    /**
     * Constructor
     */
    Executor();
    Executor(std::string model_name, int input_height, int input_width,
        int thread_num);

    /**
     * analysis and import model code
     */
    int analysisModel();

    /**
     * load image to Mat struct
     *
     * @param ai_buf: camera input image
     * @param height: height of input image
     * @param width:  width of imput image
     */
    void loadImage(uint8_t* ai_buf, int height, int width);
    void loadImage(std::string image_path, int height, int width);

    /**
     * copy data to/from ai chip
     *
     * @param host_buf: host memory pointor
     * @param ai_base: the base address of ai chip
     * @param size: memory size
     * @param dir: copy direction
     * @return error code
     */
    virtual int copyData(void* host_buf, void* ai_base, uint64_t size,
        Direction_t dir);

    /**
     * Start to inference
     */
    int compute();

    /**
     * analysis inference result
     *
     * @param result_buf: result of inference
     * @param size: result size
     * @param callback_draw_boxaram: callback to draw result
     */
    int inferenceResult(void* result_buf, uint64_t size,
        callback_draw_box call);

    /**
     * Deconstructor
     */
    ~Executor();

    Executor& operator=(const Executor& exe);

 private:
    /// thread num
    int thread_num_;
    /// image struct
    RamTensor::sptr image_ptr;
    /// pointor to graph vector
    // std::vector<Graph::ptr> graphs_ptr_;
    /// model_name
    std::string model_name_;
};

}  // namespace rvos

#endif  // INCLUDE_CORE_EXECUTOR_HPP_
