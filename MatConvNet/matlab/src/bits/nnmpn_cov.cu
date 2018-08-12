// @file  nnmpn_cov.cu
// @brief MPN-COV block
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2017 Peihua Li and Jiangtao Xie

All rights reserved.
*/


#include "nnmpn_cov.hpp"
#include "impl/nnmpn_cov_blas.hpp"
#if ENABLE_CUDNN
#include "impl/nnmpn_cov_blas.hpp" // cudnn -> blas
#endif
#include <assert.h>

#ifdef ENABLE_GPU
#include "datacu.hpp"
#endif
using namespace vl ;

#define DISPATCH(deviceType, type, dataType) \
error = vl::impl::mpn_cov<deviceType, type, dataType>::forward \
(context, \
(type*)output.getMemory(), \
(type const*)data.getMemory(), \
(type*)aux_S.getMemory(), \
(type*)aux_V.getMemory(), \
(type*)aux_D.getMemory(), \
data.getHeight(),data.getWidth(),data.getDepth(),data.getSize(), \
epsilon,alpha) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, float, VLDT_Float) ; break ; \
    IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, double, VLDT_Double) ; break ;)\
default: assert(false) ; return VLE_Unknown ; \
}

#define DISPATCHCUDNN(dataType) \
error = vl::impl::mpn_cov<deviceType, type, dataType>::forward \
(context, \
(type*)output.getMemory(), \
(type const*)data.getMemory(), \
(type*)aux_S.getMemory(), \
(type*)aux_V.getMemory(), \
(type*)aux_D.getMemory(), \
data.getHeight(),data.getWidth(),data.getDepth(),data.getSize(), \
epsilon,alpha) ;

#define DISPATCHCUDNN2() \
switch (dataType) { \
case VLDT_Float : DISPATCHCUDNN(VLDT_Float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCHCUDNN(VLDT_Double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnmpn_cov_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      vl::Tensor aux_S,
                      vl::Tensor aux_V,
                      vl::Tensor aux_D,
                      double epsilon,
                      double alpha)
{
  vl::ErrorCode error = VLE_Success ;
  vl::DataType dataType = output.getDataType() ;

  switch (output.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
//      if (context.getCudaHelper().getCudnnEnabled()) {
//        DISPATCHCUDNN2() ;
//        if (error == vl::VLE_Success) { return error ; }
//        if (error != vl::VLE_Unsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
//      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
//done:
#endif
  return context.passError(error,__func__) ;
}


#undef DISPATCH
#define DISPATCH(deviceType, type, dataType) \
error = vl::impl::mpn_cov<deviceType, type, dataType>::backward \
(context, \
 (type*)derData.getMemory(), \
 (type const*)data.getMemory(),    \
 (type const*)derOutput.getMemory(), \
 (type const*)aux_S.getMemory(), \
 (type const*)aux_V.getMemory(), \
 (type const*)aux_D.getMemory(), \
 derData.getHeight(),derData.getWidth(),derData.getDepth(),derData.getSize(), \
epsilon,alpha) ;

#undef DISPATCHCUDNN
#define DISPATCHCUDNN(dataType) \
error = vl::impl::mpn_cov<deviceType, type, dataType>::backward \
(context, \
 (type*)derData.getMemory(), \
 (type const*)data.getMemory(),    \
 (type const*)derOutput.getMemory(), \
 (type const*)aux_S.getMemory(), \
 (type const*)aux_V.getMemory(), \
 (type const*)aux_D.getMemory(), \
 derData.getHeight(),derData.getWidth(),derData.getDepth(),derData.getSize(), \
epsilon,alpha) ;

vl::ErrorCode
vl::nnmpn_cov_backward(vl::Context& context,
                       vl::Tensor derData,
                       vl::Tensor data,
                       vl::Tensor derOutput,
                       vl::Tensor aux_S,
                       vl::Tensor aux_V,
                       vl::Tensor aux_D,
                       double epsilon,
                       double alpha)
{
  vl::ErrorCode error = vl::VLE_Success ;
  vl::DataType dataType = derOutput.getDataType() ;

  switch (derOutput.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:

#if ENABLE_CUDNN
//     if (context.getCudaHelper().getCudnnEnabled()) {
//        DISPATCHCUDNN2() ;
//        if (error == vl::VLE_Success) { return error ; }
//        if (error != vl::VLE_Unsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
//      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
//done:
#endif
  return context.passError(error,__func__) ;
}

