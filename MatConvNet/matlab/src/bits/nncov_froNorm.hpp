// @file  nncov_froNorm.hpp
// @brief MPN-COV block
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2017 Peihua Li and Jiangtao Xie

All rights reserved.
*/


#ifndef __vl__nncov__froNorm__
#define __vl__nncov__froNorm__

#include "data.hpp"

namespace vl {

  vl::ErrorCode
  nncov_froNorm_forward(vl::Context& context,
                          vl::Tensor output,
                          vl::Tensor data,
					      vl::Tensor aux_T) ;

  vl::ErrorCode
  nncov_froNorm_backward(vl::Context& context,
                           vl::Tensor derData,
                           vl::Tensor data,
                           vl::Tensor derOutput,
						   vl::Tensor derOutput_aux,
					       vl::Tensor aux_T) ;
  vl::ErrorCode
  nncov_froNorm_aux_forward(vl::Context& context,   //
                              vl::Tensor output,
							  vl::Tensor data,
							  vl::Tensor aux_T);
  vl::ErrorCode
  nncov_froNorm_aux_backward(vl::Context& context,
                               vl::Tensor derData,
							   vl::Tensor derData_aux,
                               vl::Tensor data,
                               vl::Tensor derOutput,
							   vl::Tensor aux_T) ;
}


#endif /* defined(__vl__nnmpn_cov__) */
