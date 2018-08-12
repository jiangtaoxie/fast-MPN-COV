// @file blashelper_gpu.hpp
// @brief BLAS helpers implementation (CPU),
// @brief we divided the original file(blashelper.hpp) into
// @brief two parts(*_cpu and *_gpu)
// @author Jiantao Xie
// @author Peihua Li


/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__blashelper__gpu__
#define __vl__blashelper__gpu__

#include "../data.hpp"
#include "copy.hpp"
#include <blas.h>
#include <mex.h>
#include "../datacu.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>



/* ---------------------------------------------------------------- */
/* BLAS helpers                                                     */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

template<vl::DeviceType deviceType, vl::DataType dataType>
struct blas
{
  typedef typename DataTypeTraits<dataType>::type type ;

  static vl::ErrorCode
  gemm(vl::Context& context,
       char op1, char op2,
       ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * b, ptrdiff_t ldb,
       type beta,
       type * c, ptrdiff_t ldc) ;

  static vl::ErrorCode
  gemv(vl::Context& context,
       char op,
       ptrdiff_t m, ptrdiff_t n,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * x, ptrdiff_t incx,
       type beta,
       type * y, ptrdiff_t incy) ;

  static vl::ErrorCode
  axpy(vl::Context& context,
       ptrdiff_t n,
       type alpha,
       type const *x, ptrdiff_t incx,
       type *y, ptrdiff_t incy) ;

  static vl::ErrorCode
  scal(vl::Context& context,
       ptrdiff_t n,
       type alpha,
       type *x,
       ptrdiff_t inc) ;

 static vl::ErrorCode
  eig(vl::Context& context,
      type *a,
      type *w,
      type * aux_D,
      ptrdiff_t n,
      ptrdiff_t L);

  static vl::ErrorCode
  cov(Context& context,
      type const *x,
      type *P,
      type *II,
      type *lwork,
      ptrdiff_t m,
      ptrdiff_t n);
  static vl::ErrorCode
  symm(vl::Context & context,
       char side,
       ptrdiff_t m,
       ptrdiff_t n,
       type alpha,
       type const *a, ptrdiff_t lda,
       type const *b, ptrdiff_t ldb,
       type beta,
       type *c, ptrdiff_t ldc);

  static vl::ErrorCode
  dgmm(vl::Context & context,
       char side,
       ptrdiff_t m,
       ptrdiff_t n,
       type const *a, ptrdiff_t lda,
       type const *x, ptrdiff_t incx,
       type *c, int ldc);
  
  static vl::ErrorCode
  ger(vl::Context context,
      ptrdiff_t m,ptrdiff_t n,
      type const alpha,
      type const *x, ptrdiff_t incx,
      type const *y, ptrdiff_t incy,
      type *A, ptrdiff_t lda);
  static vl::ErrorCode
  dot(Context& context,
      ptrdiff_t n,
      type const *x,
      ptrdiff_t incx,
      type const *y,
      ptrdiff_t incy,
      type *result);
} ;


/* ---------------------------------------------------------------- */
/* GPU implementation                                               */
/* ---------------------------------------------------------------- */



template<>
struct blas<vl::VLDT_GPU, vl::VLDT_Float>
{
  typedef float type ;

  static vl::ErrorCode
  gemm(vl::Context& context,
       char op1, char op2,
       ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * b, ptrdiff_t ldb,
       type beta,
       type * c, ptrdiff_t ldc)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;

    status = cublasSgemm(handle,
                         (op1 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (op2 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (int)m, (int)n, (int)k,
                         &alpha,
                         a, (int)lda,
                         b, (int)ldb,
                         &beta,
                         c, (int)ldc);
  done:
    return context.setError
    (context.getCudaHelper().catchCublasError(status, "cublasSgemm"), __func__) ;
  }

  static vl::ErrorCode
  gemv(vl::Context& context,
       char op,
       ptrdiff_t m, ptrdiff_t n,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * x, ptrdiff_t incx,
       type beta,
       type * y, ptrdiff_t incy)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;
    status = cublasSgemv(handle,
                         (op == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (int)m, (int)n,
                         &alpha,
                         a, lda,
                         x, (int)incx,
                         &beta,
                         y, (int)incy);

  done:
    return context.setError
      (context.getCudaHelper().catchCublasError(status, "cublasSgemv"), __func__) ;
  }

  static vl::ErrorCode
  scal(vl::Context & context,
       ptrdiff_t n,
       type alpha,
       type *x, ptrdiff_t incx)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;
    status = cublasSscal(handle,
                         (int)n,
                         &alpha,
                         x, (int)incx) ;
  done:
    return context.setError
      (context.getCudaHelper().catchCublasError(status, "cublasSscal"), __func__) ;
  }

  static vl::ErrorCode
  axpy(vl::Context & context,
       ptrdiff_t n,
       type alpha,
       type const *x, ptrdiff_t incx,
       type *y, ptrdiff_t incy)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;
    status = cublasSaxpy(handle,
                         (int)n,
                         &alpha,
                         x, (int)incx,
                         y, (int)incy) ;
  done:
    return context.setError
      (context.getCudaHelper().catchCublasError(status, "cublasSaxpy"), __func__) ;
  }

  static vl::ErrorCode
  eig(vl::Context& context,
      type *a,
      type *w,
      type * aux_D,
      ptrdiff_t n,
      ptrdiff_t L)
  {
      vl::ErrorCode error = VLE_Success;
      mxArray *lhs[3],*x;
      mxClassID classID = mxSINGLE_CLASS ;
      mwSize dim[3];
      dim[0] = n;dim[1] = n;dim[2] = L;
      x  = mxCreateNumericArray(3,dim,classID,mxREAL);
      type *xptr = (type*)mxGetData(x);
      type *s;
      type *v;
      type *Dmin;
      cudaMemcpy(xptr,a,sizeof(type)*n*n*L,cudaMemcpyDeviceToHost);
      mexCallMATLAB(3,lhs,1,&x,"EIG");
      v    = (type*)mxGetData(lhs[0]);
      s    = (type*)mxGetData(lhs[1]);
      Dmin = (type*)mxGetData(lhs[2]);
      memcpy(aux_D,Dmin,sizeof(type)*L); 
      cudaMemcpy(a,v,sizeof(type)*n*n*L,cudaMemcpyHostToDevice);
      cudaMemcpy(w,s,sizeof(type)*n*L,cudaMemcpyHostToDevice);
      mxDestroyArray(lhs[0]);
      mxDestroyArray(lhs[1]);
      mxDestroyArray(lhs[2]);
      mxDestroyArray(x);
done:
      return error;
  }

  static vl::ErrorCode
  cov(Context& context,
      type const *x,
      type *P,
      type *II,
      type *lwork,
      ptrdiff_t m,
      ptrdiff_t n)
  {
      vl::ErrorCode error;
     error = vl::impl::blas<vl::VLDT_GPU , vl::VLDT_Float>::gemm(context,
                                                                't','n',
                                                                n,m,m,
                                                                (type)1,
                                                                x,m,
                                                                II,m,
                                                                (type)0,
                                                                lwork,n);
     if (error != vl::VLE_Success) {
         goto done ;
     }
     error = vl::impl::blas<vl::VLDT_GPU , vl::VLDT_Float>::gemm(context,
                                                                'n','n',
                                                                n,n,m,
                                                                (type)1,
                                                                lwork,n,
                                                                x,m,
                                                                (type)0,
                                                                P,n);
     if (error != vl::VLE_Success){
         goto done ;
     }
done:
     return error;
  }

  static vl::ErrorCode
  symm(vl::Context & context,
       char side,
       ptrdiff_t m,
       ptrdiff_t n,
       type alpha,
       type const *a, ptrdiff_t lda,
       type const *b, ptrdiff_t ldb,
       type beta,
       type *c, ptrdiff_t ldc)
  {
      cublasHandle_t handle ;
      cublasStatus_t status ;
      status = context.getCudaHelper().getCublasHandle(&handle) ;
      if (status != CUBLAS_STATUS_SUCCESS) goto done ;
      status = cublasSsymm(handle,
                           (side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                           CUBLAS_FILL_MODE_UPPER,
                           (int)m, (int)n,
                           &alpha,
                           a, (int)lda,
                           b, (int)ldb,
                           &beta,
                           c, (int)ldc);
      done:
        return context.setError
        (context.getCudaHelper().catchCublasError(status, "cublasSsymm"), __func__) ;
  }

  static vl::ErrorCode
  dgmm(vl::Context & context,
       char side,
       ptrdiff_t m,
       ptrdiff_t n,
       type const *a, ptrdiff_t lda,
       type const *x, ptrdiff_t incx,
       type *c, int ldc)
  {
      cublasHandle_t handle ;
      cublasStatus_t status ;
      status = context.getCudaHelper().getCublasHandle(&handle) ;
      if (status != CUBLAS_STATUS_SUCCESS) goto done ;
      status = cublasSdgmm(handle,
                           (side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                           m,n,
                           a, (int)lda,
                           x,  (int)incx,
                           c, (int)ldc);
      done:
        return context.setError
        (context.getCudaHelper().catchCublasError(status, "cublasSdgmm"), __func__) ;
  }

  static vl::ErrorCode
  ger(vl::Context context,
      ptrdiff_t m,ptrdiff_t n,
      type const alpha,
      type const *x, ptrdiff_t incx,
      type const *y, ptrdiff_t incy,
      type *A, ptrdiff_t lda)
  {
      cublasHandle_t handle ;
      cublasStatus_t status ;
      status = context.getCudaHelper().getCublasHandle(&handle) ;
      if (status != CUBLAS_STATUS_SUCCESS) goto done ;
      status = cublasSger(handle,
                          (int)m,(int)n,
                          &alpha,
                          x, (int)incx,
                          y, (int)incy,
                          A, lda);
       done:
        return context.setError
        (context.getCudaHelper().catchCublasError(status, "cublasSger"), __func__) ;
  }
  static vl::ErrorCode
  dot(Context& context,
      ptrdiff_t n,
      type const *x,
      ptrdiff_t incx,
      type const *y,
      ptrdiff_t incy,
      type *result)
  {
	  cublasHandle_t handle ;
      cublasStatus_t status ;
      status = context.getCudaHelper().getCublasHandle(&handle) ;
      if (status != CUBLAS_STATUS_SUCCESS) goto done ;
	  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	  status = cublasSdot(handle,
		                  (int)n,
						  x, (int)incx,
						  y, (int)incy,
						  result);
	   done:
        return context.setError
        (context.getCudaHelper().catchCublasError(status, "cublasSdot"), __func__) ;
  }

} ;

template<>
struct blas<vl::VLDT_GPU, vl::VLDT_Double>
{
  typedef double type ;

  static vl::ErrorCode
  gemm(vl::Context& context,
       char op1, char op2,
       ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * b, ptrdiff_t ldb,
       type beta,
       type * c, ptrdiff_t ldc)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;

    status = cublasDgemm(handle,
                         (op1 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (op2 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (int)m, (int)n, (int)k,
                         &alpha,
                         a, (int)lda,
                         b, (int)ldb,
                         &beta,
                         c, (int)ldc);
  done:
    return context.setError
    (context.getCudaHelper().catchCublasError(status, "cublasDgemm"), __func__) ;
  }

  static vl::ErrorCode
  gemv(vl::Context& context,
       char op,
       ptrdiff_t m, ptrdiff_t n,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * x, ptrdiff_t incx,
       type beta,
       type * y, ptrdiff_t incy)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;

    status = cublasDgemv(handle,
                         (op == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (int)m, (int)n,
                         &alpha,
                         a, lda,
                         x, (int)incx,
                         &beta,
                         y, (int)incy);
  done:
    return context.setError
    (context.getCudaHelper().catchCublasError(status, "cublasDgemv"), __func__) ;
  }

  static vl::ErrorCode
  scal(vl::Context & context,
       ptrdiff_t n,
       type alpha,
       type *x, ptrdiff_t incx)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;
    status = cublasDscal(handle,
                         (int)n,
                         &alpha,
                         x, (int)incx) ;
  done:
    return context.setError
      (context.getCudaHelper().catchCublasError(status, "cublasDscal"), __func__) ;
  }

  static vl::ErrorCode
  axpy(vl::Context & context,
       ptrdiff_t n,
       type alpha,
       type const *x, ptrdiff_t incx,
       type *y, ptrdiff_t incy)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;
    status = cublasDaxpy(handle,
                         (int)n,
                         &alpha,
                         x, (int)incx,
                         y, (int)incy) ;
  done:
    return context.setError
      (context.getCudaHelper().catchCublasError(status, "cublasDaxpy"), __func__) ;
  }
static vl::ErrorCode
  eig(vl::Context& context,
      type *a,
      type *w,
      type * aux_D,
      ptrdiff_t n,
      ptrdiff_t L)
  {
      vl::ErrorCode error = VLE_Success;
      mxArray *lhs[3],*x;
      mxClassID classID = mxDOUBLE_CLASS ;
      mwSize dim[3];
      dim[0] = n;dim[1] = n;dim[2] = L;
      x  = mxCreateNumericArray(3,dim,classID,mxREAL);
      type *xptr = (type*)mxGetData(x);
      type *s;
      type *v;
      type *Dmin;
      cudaMemcpy(xptr,a,sizeof(type)*n*n*L,cudaMemcpyDeviceToHost);
      mexCallMATLAB(3,lhs,1,&x,"EIG");
      v    = (type*)mxGetData(lhs[0]);
      s    = (type*)mxGetData(lhs[1]);
      Dmin = (type*)mxGetData(lhs[2]);
      memcpy(aux_D,Dmin,sizeof(type)*L); 
      cudaMemcpy(a,v,sizeof(type)*n*n*L,cudaMemcpyHostToDevice);
      cudaMemcpy(w,s,sizeof(type)*n*L,cudaMemcpyHostToDevice);
      mxDestroyArray(lhs[0]);
      mxDestroyArray(lhs[1]);
      mxDestroyArray(lhs[2]);
      mxDestroyArray(x);
done:
      return error;
  }

    static vl::ErrorCode
  cov(Context& context,
      type const *x,
      type *P,
      type *II,
      type *lwork,
      ptrdiff_t m,
      ptrdiff_t n)
  {
     vl::ErrorCode error;
     error = vl::impl::blas<vl::VLDT_GPU , vl::VLDT_Double>::gemm(context,
                                                                't','n',
                                                                n,m,m,
                                                                (type)1,
                                                                x,m,
                                                                II,m,
                                                                (type)0,
                                                                lwork,n);
     if (error != vl::VLE_Success) {
         goto done ;
     }
     error = vl::impl::blas<vl::VLDT_GPU , vl::VLDT_Double>::gemm(context,
                                                                'n','n',
                                                                n,n,m,
                                                                (type)1,
                                                                lwork,n,
                                                                x,m,
                                                                (type)0,
                                                                P,n);
     if (error != vl::VLE_Success){
         goto done ;
     }
done:
     return error;
  }
    static vl::ErrorCode
  symm(vl::Context & context,
       char side,
       ptrdiff_t m,
       ptrdiff_t n,
       type alpha,
       type const *a, ptrdiff_t lda,
       type const *b, ptrdiff_t ldb,
       type beta,
       type *c, ptrdiff_t ldc)
  {
      cublasHandle_t handle ;
      cublasStatus_t status ;
      status = context.getCudaHelper().getCublasHandle(&handle) ;
      if (status != CUBLAS_STATUS_SUCCESS) goto done ;
      status = cublasDsymm(handle,
                           (side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                           CUBLAS_FILL_MODE_UPPER,
                           (int)m, (int)n,
                           &alpha,
                           a, (int)lda,
                           b, (int)ldb,
                           &beta,
                           c, (int)ldc);
      done:
        return context.setError
        (context.getCudaHelper().catchCublasError(status, "cublasDsymm"), __func__) ;
  }
    static vl::ErrorCode
  dgmm(vl::Context & context,
       char side,
       ptrdiff_t m,
       ptrdiff_t n,
       type const *a, ptrdiff_t lda,
       type const *x, ptrdiff_t incx,
       type *c, int ldc)
  {
      cublasHandle_t handle ;
      cublasStatus_t status ;
      status = context.getCudaHelper().getCublasHandle(&handle) ;
      if (status != CUBLAS_STATUS_SUCCESS) goto done ;
      status = cublasDdgmm(handle,
                           (side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                           m,n,
                           a, (int)lda,
                           x,  (int)incx,
                           c, (int)ldc);
      done:
        return context.setError
        (context.getCudaHelper().catchCublasError(status, "cublasDdgmm"), __func__) ;
  }

  static vl::ErrorCode
  ger(vl::Context context,
      ptrdiff_t m,ptrdiff_t n,
      type const alpha,
      type const *x, ptrdiff_t incx,
      type const *y, ptrdiff_t incy,
      type *A, ptrdiff_t lda)
  {
      cublasHandle_t handle ;
      cublasStatus_t status ;
      status = context.getCudaHelper().getCublasHandle(&handle) ;
      if (status != CUBLAS_STATUS_SUCCESS) goto done ;
      status = cublasDger(handle,
                          (int)m,(int)n,
                          &alpha,
                          x, (int)incx,
                          y, (int)incy,
                          A, lda);
       done:
        return context.setError
        (context.getCudaHelper().catchCublasError(status, "cublasDger"), __func__) ;
  }
   static vl::ErrorCode
  dot(Context& context,
      ptrdiff_t n,
      type const *x,
      ptrdiff_t incx,
      type const *y,
      ptrdiff_t incy,
      type *result)
  {
	  cublasHandle_t handle ;
      cublasStatus_t status ;
      status = context.getCudaHelper().getCublasHandle(&handle) ;
      if (status != CUBLAS_STATUS_SUCCESS) goto done ;
	  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	  status = cublasDdot(handle,
		                  (int)n,
						  x, (int)incx,
						  y, (int)incy,
						  result);
	   done:
        return context.setError
        (context.getCudaHelper().catchCublasError(status, "cublasSdot"), __func__) ;
  }

} ;


} } // namespace vl { namespace impl {
#endif /* defined(__vl__blashelper__) */
