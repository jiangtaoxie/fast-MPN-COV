// @file  nnmpn_cov.hpp
// @brief MPN-COV block
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2017 Peihua Li and Jiangtao Xie

All rights reserved.
*/


#ifndef __vl__nnmpn__cov__
#define __vl__nnmpn__cov__

#include "data.hpp"

namespace vl {

  vl::ErrorCode
  nnmpn_cov_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor data,
                    vl::Tensor aux_S,
                    vl::Tensor aux_V,
                    vl::Tensor aux_D,
                    double epsilon,
                    double alpha) ;

  vl::ErrorCode
  nnmpn_cov_backward(vl::Context& context,
                     vl::Tensor derData,
                     vl::Tensor data,
                     vl::Tensor derOutput,
                     vl::Tensor aux_S,
                     vl::Tensor aux_V,
                     vl::Tensor aux_D,
                     double epsilon,
                     double alpha) ;
}


#endif /* defined(__vl__nnmpn_cov__) */
