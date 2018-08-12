// @file  nncov_traceNorm.hpp
// @brief MPN-COV block
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2017 Peihua Li and Jiangtao Xie

All rights reserved.
*/


#ifndef __vl__nncov__traceNorm__
#define __vl__nncov__traceNorm__

#include "data.hpp"

namespace vl {

  vl::ErrorCode
  nncov_traceNorm_forward(vl::Context& context,
                          vl::Tensor output,
                          vl::Tensor data,
					      vl::Tensor aux_T) ;

  vl::ErrorCode
  nncov_traceNorm_backward(vl::Context& context,
                           vl::Tensor derData,
                           vl::Tensor data,
                           vl::Tensor derOutput,
						   vl::Tensor derOutput_aux,
					       vl::Tensor aux_T) ;
  vl::ErrorCode
  nncov_traceNorm_aux_forward(vl::Context& context,   //
                              vl::Tensor output,
							  vl::Tensor data,
							  vl::Tensor aux_T);
  vl::ErrorCode
  nncov_traceNorm_aux_backward(vl::Context& context,
                               vl::Tensor derData,
							   vl::Tensor derData_aux,
                               vl::Tensor data,
                               vl::Tensor derOutput,
							   vl::Tensor aux_T) ;
}


#endif /* defined(__vl__nnmpn_cov__) */
