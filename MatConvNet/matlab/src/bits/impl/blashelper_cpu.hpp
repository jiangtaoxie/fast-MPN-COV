// @file blashelper_cpu.hpp
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

#ifndef __vl__blashelper__cpu__
#define __vl__blashelper__cpu__

#include "../data.hpp"
#include "copy.hpp"
#include <blas.h>
#include <mex.h>
#include <memory.h>
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
      type *aux_D,
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
  dot(Context& context,
      ptrdiff_t n,
      type const *x,
      ptrdiff_t incx,
      type const *y,
      ptrdiff_t incy,
      type *result);
} ;

/* ---------------------------------------------------------------- */
/* CPU implementation                                               */
/* ---------------------------------------------------------------- */

template<>
struct blas<vl::VLDT_CPU, vl::VLDT_Float>
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
    sgemm(&op1, &op2,
          &m, &n, &k,
          &alpha,
          (type*)a, &lda,
          (type*)b, &ldb,
          &beta,
          c, &ldc) ;
    return vl::VLE_Success ;
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
    sgemv(&op,
          &m, &n, &alpha,
          (float*)a, &lda,
          (float*)x, &incx,
          &beta,
          y, &incy) ;
    return vl::VLE_Success ;
  }

  static vl::ErrorCode
  axpy(vl::Context & context,
       ptrdiff_t n,
       type alpha,
       type const *x, ptrdiff_t incx,
       type *y, ptrdiff_t incy)
  {
    saxpy(&n,
          &alpha,
          (float*)x, &incx,
          (float*)y, &incy) ;
    return vl::VLE_Success ;
  }

  static vl::ErrorCode
  scal(vl::Context & context,
       ptrdiff_t n,
       type alpha,
       type *x, ptrdiff_t incx)
  {
    sscal(&n,
          &alpha,
          (float*)x, &incx) ;
    return vl::VLE_Success ;
  }

 static vl::ErrorCode
  eig(vl::Context& context,
      type *a,
      type *w,
      type *aux_D,
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
      memcpy(xptr,a,sizeof(type)*n*n*L);
      mexCallMATLAB(3,lhs,1,&x,"EIG");
      v    = (type*)mxGetData(lhs[0]);
      s    = (type*)mxGetData(lhs[1]);
      Dmin = (type*)mxGetData(lhs[2]);
      memcpy(w,s,sizeof(type)*n*L);
      memcpy(a,v,sizeof(type)*n*n*L);
      memcpy(aux_D,Dmin,sizeof(type)*L);
      mxDestroyArray(lhs[0]);
      mxDestroyArray(lhs[1]);
      mxDestroyArray(lhs[2]);
      mxDestroyArray(x);
      return vl::VLE_Success;
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
    error = vl::impl::blas<vl::VLDT_CPU , vl::VLDT_Float>::gemm(context,
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
    error = vl::impl::blas<vl::VLDT_CPU , vl::VLDT_Float>::gemm(context,
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
  dot(Context& context,
      ptrdiff_t n,
      type const *x,
      ptrdiff_t incx,
      type const *y,
      ptrdiff_t incy,
      type *result)
  {
	  *result = sdot(&n,
		             x,&incx,
		             y,&incy);
	  return vl::VLE_Success;
  }

} ;

template<>
struct blas<vl::VLDT_CPU, vl::VLDT_Double>
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
    dgemm(&op1, &op2,
          &m, &n, &k,
          &alpha,
          (type*)a, &lda,
          (type*)b, &ldb,
          &beta,
          c, &ldc) ;
    return vl::VLE_Success ;
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
    dgemv(&op,
          &m, &n, &alpha,
          (type*)a, &lda,
          (type*)x, &incx,
          &beta,
          y, &incy) ;
    return vl::VLE_Success ;
  }

  static vl::ErrorCode
  axpy(vl::Context & context,
       ptrdiff_t n,
       type alpha,
       type const *x, ptrdiff_t incx,
       type *y, ptrdiff_t incy)
  {
    daxpy(&n,
          &alpha,
          (double*)x, &incx,
          (double*)y, &incy) ;
    return vl::VLE_Success ;
  }

  static vl::ErrorCode
  scal(vl::Context & context,
       ptrdiff_t n,
       type alpha,
       type *x, ptrdiff_t incx)
  {
    dscal(&n,
          &alpha,
          (double*)x, &incx) ;
    return vl::VLE_Success ;
  }

  static vl::ErrorCode
  eig(vl::Context& context,
      type *a,
      type *w,
      type *aux_D,
      ptrdiff_t n,
      ptrdiff_t L)
  {
      vl::ErrorCode error = VLE_Success;
      mxArray *lhs[2],*x;
      mxClassID classID = mxDOUBLE_CLASS ;
      mwSize dim[3];
      dim[0] = n;dim[1] = n;dim[2] = L;
      x  = mxCreateNumericArray(3,dim,classID,mxREAL);
      type *xptr = (type*)mxGetData(x);
      type *s;
      type *v;
      type *Dmin;
      memcpy(xptr,a,sizeof(type)*n*n*L);
      mexCallMATLAB(3,lhs,1,&x,"EIG");
      v    = (type*)mxGetData(lhs[0]);
      s    = (type*)mxGetData(lhs[1]);
      Dmin = (type*)mxGetData(lhs[2]);
      memcpy(w,s,sizeof(type)*n*L);
      memcpy(a,v,sizeof(type)*n*n*L);
      memcpy(aux_D,Dmin,sizeof(type)*L);
      mxDestroyArray(lhs[0]);
      mxDestroyArray(lhs[1]);
      mxDestroyArray(lhs[2]);
      mxDestroyArray(x);
      return vl::VLE_Success;
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
    error = vl::impl::blas<vl::VLDT_CPU , vl::VLDT_Double>::gemm(context,
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
    error = vl::impl::blas<vl::VLDT_CPU , vl::VLDT_Double>::gemm(context,
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
  dot(Context& context,
      ptrdiff_t n,
      type const *x,
      ptrdiff_t incx,
      type const *y,
      ptrdiff_t incy,
      type *result)
  {
	  *result = ddot(&n,
		             x,&incx,
		             y,&incy);
	  return vl::VLE_Success;
  }
} ;



} } // namespace vl { namespace impl {
#endif /* defined(__vl__blashelper__) */
