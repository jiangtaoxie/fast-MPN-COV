// @file  nncov_sqrtm_blas.hpp
// @brief fast MPN-COV header
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
*/

#ifndef __vl__nncov__sqrtm__blas__
#define __vl__nncov__sqrtm__blas__

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {
    template<vl::DeviceType dev ,typename type,vl::DataType dataType>
    struct cov_sqrtm
    {
        static vl::ErrorCode
            forward(Context& context,
                    type* output,
                    type const* data,
                    type* aux_Y,
                    type* aux_Z,
                    size_t height, size_t width, size_t depth, size_t num,
                    int coef,
                    int iterNum);
        static vl::ErrorCode
            backward(Context& context,
                     type* derData,
                     type const* data,
                     type const* derOutput,
                     type const* aux_Y,
                     type const* aux_Z,
                     size_t height, size_t width, size_t depth, size_t num,
                     int coef,
                     int iterNum);
    };
} }


#endif /* __vl_cov_sqrtm__ */
