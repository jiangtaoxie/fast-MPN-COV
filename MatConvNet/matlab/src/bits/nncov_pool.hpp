// @file  nncov_pool.hpp
// @brief MPN-COV block
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2017 Peihua Li and Jiangtao Xie

All rights reserved.
*/


#ifndef __vl__nncov__pool__
#define __vl__nncov__pool__

#include "data.hpp"

namespace vl {

  vl::ErrorCode
  nncov_pool_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor data) ;

  vl::ErrorCode
  nncov_pool_backward(vl::Context& context,
                     vl::Tensor derData,
                     vl::Tensor data,
                     vl::Tensor derOutput) ;
}


#endif /* defined(__vl__nnmpn_cov__) */
