// @file  nncov_sqrtm.hpp
// @brief MPN-COV block
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2017 Peihua Li and Jiangtao Xie

All rights reserved.
*/


#ifndef __vl__nncov__sqrtm__
#define __vl__nncov__sqrtm__

#include "data.hpp"

namespace vl {

  vl::ErrorCode
  nncov_sqrtm_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      vl::Tensor aux_Y,
                      vl::Tensor aux_Z,
					  int coef,
                      int iterNum) ;

  vl::ErrorCode
  nncov_sqrtm_backward(vl::Context& context,
                       vl::Tensor derData,
                       vl::Tensor data,
                       vl::Tensor derOutput,
                       vl::Tensor aux_Y,
                       vl::Tensor aux_Z,
					   int coef,
                       int iterNum) ;
}


#endif /* defined(__vl__nnmpn_cov__) */
