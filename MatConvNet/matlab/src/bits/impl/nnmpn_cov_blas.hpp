// @file  nnmpn_cov_blas.hpp
// @brief MPN-COV 
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2017 Peihua Li and Jiangtao Xie

All rights reserved.
*/

#ifndef __vl__nnmpn__cov__blas__
#define __vl__nnmpn__cov__blas__

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {
    template<vl::DeviceType dev ,typename type,vl::DataType dataType>
    struct mpn_cov
    {
        static vl::ErrorCode
            forward(Context& context,
                    type* output,
                    type const* data,
                    type* aux_S,
                    type* aux_V,
                    type* aux_D,
                    size_t height, size_t width, size_t depth, size_t num,
                    type epsilon,
                    type alpha);
        static vl::ErrorCode
            backward(Context& context,
                     type* derData,
                     type const* data,
                     type const* derOutput,
                     type const* aux_S,
                     type const* aux_V,
                     type const* aux_D,
                     size_t height, size_t width, size_t depth, size_t num,
                     type epsilon,
                     type alpha);
    };
} }


#endif /* __vl_mpn_cov__ */
