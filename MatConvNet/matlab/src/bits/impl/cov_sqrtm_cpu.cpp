// @file  cov_sqrtm_gpu.cpp
// @brief fast MPN-COV implementation (CPU)
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
*/
#include "nncov_sqrtm_blas.hpp"
#include "../data.hpp"
#include <math.h>
#include <memory>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <cassert>
#include "blashelper_cpu.hpp"




template<typename T> inline void
getOutput_cpu(T* output,
          T* result,
          ptrdiff_t n)
{
    int i,offset = 0;
    for(i = 0; i < n; i ++){
        offset += i; 
        vl::impl::operations<vl::VLDT_CPU, T>::copy(output + offset,result + i*n,i+1);
    }
}

template<typename T> inline void
getdLdCfromderOutput_cpu(T* dLdC,
                         T const* derOutput,
                         ptrdiff_t n)
{
    int i,offset = 0;
    for(i = 0;i < n;i ++){
        offset += i;
        vl::impl::operations<vl::VLDT_CPU, T>::copy(dLdC + i*n,derOutput + offset,i + 1);
    }
}


template<typename T> inline void
symmetric_cpu(T* a,
              ptrdiff_t n)
{
    int lda = (int)n;
    int offset = 0;
    int i,j;
    for(i = 0;i < n;i ++) {
        offset = i;
        for(j = offset;j < n;j ++) {
            a[i * lda + j] = (a[i * lda + j] + a[j * lda + i]) / 2.0f;
            a[j * lda + i] = a[i * lda + j];
        }
    }
}
template<typename T> inline void
matrixAdd_cpu(T* x,
              T* y,
              T* z,
              T alpha,
              T beta,
              T sigma,
              ptrdiff_t n)
{
    int i;
    for(i = 0;i < n;i++){
        x[i] = alpha * x[i] + beta * y[i] + sigma * z[i];
    }
}
namespace vl { namespace impl {
    template<typename T,vl::DataType dataType>
    struct cov_sqrtm<vl::VLDT_CPU,T,dataType>
    {
        static vl::ErrorCode
            forward(Context& context,
                    T* output,
                    T const* data,
                    T* aux_Y,
                    T* aux_Z,
                    size_t height, size_t width, size_t depth, size_t num,
                    int coef,
                    int iterNum)
        {
            vl::ErrorCode error;
            ptrdiff_t m = height,n = width,L = num,d,i;
            ptrdiff_t dataOffset,ypOffset;
            ptrdiff_t aux_YOffset,aux_ZOffset;
            ptrdiff_t aux_YOffset_1,aux_ZOffset_1,outputOffset;
            unsigned int workspaceSize =  (unsigned int)(n*n*2);
            T* workspace = (T*)context.getWorkspace(vl::VLDT_CPU , workspaceSize*sizeof(T));
            T* I3        = workspace;
            T* result    = I3 + n*n;

            T* ZY        = NULL;
            vl::impl::operations<vl::VLDT_CPU, T>::fill(I3, n*n, T(0));
            for(d = 0;d < n;d++){
                I3[d*(n+1)] = 3;
            }
            if (iterNum < 2){
                for(d = 0;d < L;d ++){
                    ZY    = aux_Z + n*n*iterNum*d; // Z0
                    dataOffset = d*n*n; 
                    aux_YOffset = n*n*iterNum*d; //Y0
                    outputOffset = d*n*(n+1)/2;
                    memcpy(ZY, data + dataOffset,sizeof(T)*n*n);
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::axpy(context,
                                                                         n*n,(T)(-1),
                                                                         I3,ptrdiff_t(1),
                                                                         ZY,ptrdiff_t(1));
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                         n,n,n,
                                                                         (T)(-0.5),data + dataOffset,n,
                                                                         ZY,n,
                                                                         (T)0,aux_Y + aux_YOffset,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    getOutput_cpu(output + outputOffset,aux_Y + aux_YOffset,n);
                    error = vl::VLE_Success;
                }
            } else {
                for(d = 0;d < L; d++){
                    ZY    = aux_Z + n*n*iterNum*d + n*n; // Z1
                    dataOffset = d*n*n; 
                    aux_YOffset = n*n*iterNum*d + n*n; //Y1
                    ypOffset = d*(n*n*iterNum);
                    outputOffset = d*n*(n+1)/2;
                    memcpy(ZY, data + dataOffset,sizeof(T)*n*n);
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::axpy(context,
                                                                         n*n,(T)(-1),
                                                                         I3,ptrdiff_t(1),
                                                                         ZY,ptrdiff_t(1));
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                         n,n,n,
                                                                         (T)(-0.5),data + dataOffset,n,
                                                                         ZY,n,
                                                                         (T)0,aux_Y + aux_YOffset,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::scal(context,
                                                                         n*n,
                                                                         (T)(-0.5),
                                                                         ZY,ptrdiff_t(1));
                    if(error != vl::VLE_Success) {goto done ;}
                    for(i = 2;i < iterNum;i++){
                        ZY = aux_Z + n*n*iterNum*d;
                        aux_YOffset = n*n*iterNum*d + i*n*n;
                        aux_ZOffset = n*n*iterNum*d + i*n*n;
                        aux_YOffset_1 = n*n*iterNum*d + (i-1)*n*n;
                        aux_ZOffset_1 = n*n*iterNum*d + (i-1)*n*n;
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                             n,n,n,
                                                                             (T)1,aux_Z + aux_ZOffset_1,n,
                                                                             aux_Y + aux_YOffset_1,n,
                                                                             (T)0,ZY,n);
                        if(error != vl::VLE_Success) {goto done ;}
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::axpy(context,
                                                                             n*n,(T)(-1),
                                                                             I3,ptrdiff_t(1),
                                                                             ZY,ptrdiff_t(1));
                        if(error != vl::VLE_Success) {goto done ;}
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                             n,n,n,
                                                                             (T)(-0.5),aux_Y + aux_YOffset_1,n,
                                                                             ZY,n,
                                                                             (T)0,aux_Y + aux_YOffset,n);
                        if(error != vl::VLE_Success) {goto done ;}
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                             n,n,n,
                                                                             (T)(-0.5),ZY,n,
                                                                             aux_Z + aux_ZOffset_1,n,
                                                                             (T)0,aux_Z + aux_ZOffset,n);
                        if(error != vl::VLE_Success) {goto done ;}
                    }
                    ZY = aux_Z + n*n*iterNum*d;
                    aux_YOffset = n*n*iterNum*d + (i-1)*n*n;
                    aux_ZOffset = n*n*iterNum*d + (i-1)*n*n;
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                         n,n,n,
                                                                         (T)1,aux_Z + aux_ZOffset,n,
                                                                         aux_Y + aux_YOffset,n,
                                                                         (T)0,ZY,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::axpy(context,
                                                                         n*n,(T)(-1),
                                                                         I3,ptrdiff_t(1),
                                                                         ZY,ptrdiff_t(1));
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                         n,n,n,
                                                                         (T)(-0.5*(T)(coef)),aux_Y + aux_YOffset,n,
                                                                         ZY,n,
                                                                         (T)0,result,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    getOutput_cpu(output + outputOffset,result,n);
                }
            }

        done:
            return context.passError(error, __func__);
        }
        static vl::ErrorCode
            backward(Context& context,
                     T* derData,
                     T const* data,
                     T const* derOutput,
                     T const* aux_Y,
                     T const* aux_Z,
                     size_t height, size_t width, size_t depth, size_t num,
                     int coef,
                     int iterNum)
        {
            vl::ErrorCode error;
            ptrdiff_t m = height,n = width,L = num,d,i;
            ptrdiff_t derOutputOffset,dLdCOffset,dataOffset;
            ptrdiff_t aux_YOffset,aux_ZOffset,aux_TOffset;
            ptrdiff_t aux_YOffset_1,aux_ZOffset_1;
            ptrdiff_t dLdYOffset,dLdZOffset;
            ptrdiff_t dLdYOffset_1,dLdZOffset_1;
            ptrdiff_t derDataOffset;
            unsigned int workspaceSize =  (unsigned int)(n*n*(iterNum)*2 + n*n*L + n*n*7);
            T* workspace = (T*)context.getWorkspace(vl::VLDT_CPU , workspaceSize*sizeof(T));
            T* dLdY      = workspace;
            T* dLdZ      = workspace + n*n*(iterNum);
            T* I3        = dLdZ + n*n*(iterNum);
            T* dLdC      = I3 + n*n;
            T* iterMemA  = dLdC + n*n*L;
            T* iterMemB  = iterMemA + n*n;
            T* iterMemC  = iterMemB + n*n;
            T* ZY        = iterMemC + n*n;
            T* YZ        = ZY + n*n;
            T* ZY_dLdY   = NULL;T* dLdZ_ZY   = NULL;
            T* Z_dLdZ    = NULL;T* Y_dLdY    = NULL;
            T* Z_dLdZ_Z  = NULL;T* Y_dLdY_Y  = NULL;
            T* dLdX      = NULL;T* dLdP      = NULL;
            T  const* P  = NULL;
            T  P_dot_dLdP;
            vl::impl::operations<vl::VLDT_CPU, T>::fill(dLdC, n*n*L , (T)0);
            for(d = 0;d < n;d++){
                I3[d*(n+1)] = 3;
            }
            for(d = 0;d < L;d++){
                derOutputOffset = d*n*(n+1)/2;
                dLdCOffset      = d*n*n;
                getdLdCfromderOutput_cpu(dLdC + dLdCOffset, derOutput + derOutputOffset, n);
                //symmetric_cpu(dLdC + derOutputOffset,n);
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::scal(context,
                                                                     n*n,
                                                                     (T)(coef),
                                                                     dLdC + dLdCOffset,ptrdiff_t(1));
                if(error != vl::VLE_Success) {goto done ;}
            }
            if (iterNum < 2) {
                for(d = 0; d < L;d++){
                    derDataOffset = d*m*n;
                    dLdP  = derData + derDataOffset;
                    dLdCOffset      = d*n*n;
                    dataOffset  = d*m*n; 
                    P     = data + dataOffset;
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                          n,n,n,
                                                                          (T)(-0.5),dLdC + dLdCOffset,n,
                                                                          P,n,
                                                                          (T)0,dLdP,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                         n,n,n,
                                                                         (T)(-0.5),P,n,
                                                                         dLdC + dLdCOffset,n,
                                                                         (T)(1),dLdP,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    matrixAdd_cpu(dLdP,dLdC + dLdCOffset,dLdP,T(1),T(1.5),T(0),n*n);
                }
            } else {
                for(d = 0;d < L;d++){
                    derDataOffset = d*m*n;
                    aux_TOffset = d;
                    aux_YOffset = n*n*iterNum*d + n*n*(iterNum - 1);
                    aux_ZOffset = n*n*iterNum*d + n*n*(iterNum - 1);
                    dLdYOffset  = n*n*(iterNum - 1);
                    dLdZOffset  = n*n*(iterNum - 1);
                    dataOffset  = d*m*n; 
                    dLdCOffset  = d*n*n;
                    ZY_dLdY     = iterMemA;
                    Y_dLdY      = iterMemC;
                    P           = data + dataOffset;
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                          n,n,n,
                                                                         (T)1,aux_Y + aux_YOffset,n,
                                                                          aux_Z + aux_ZOffset,n,
                                                                         (T)0,YZ,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::axpy(context,
                                                                         n*n,(T)(-1),
                                                                         I3,ptrdiff_t(1),
                                                                         YZ,ptrdiff_t(1));
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                         n,n,n,
                                                                         (T)1,aux_Z + aux_ZOffset,n,
                                                                         aux_Y + aux_YOffset,n,
                                                                         (T)0,ZY,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                         n,n,n,
                                                                         (T)(-0.5),ZY,n,
                                                                         dLdC + dLdCOffset,n,
                                                                         (T)0,ZY_dLdY,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                         n,n,n,
                                                                         (T)(-0.5),dLdC + dLdCOffset,n,
                                                                          YZ,n,
                                                                         (T)0,dLdY + dLdYOffset,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::axpy(context,
                                                                         n*n,(T)1,
                                                                         ZY_dLdY,ptrdiff_t(1),
                                                                         dLdY + dLdYOffset,ptrdiff_t(1));
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                         n,n,n,
                                                                         (T)1,aux_Y + aux_YOffset,n,
                                                                         dLdC + dLdCOffset,n,
                                                                         (T)0,Y_dLdY,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                          n,n,n,
                                                                          (T)(-0.5),Y_dLdY,n,
                                                                          aux_Y + aux_YOffset,n,
                                                                          (T)0,dLdZ + dLdZOffset,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    for(i = iterNum - 1;i > 1;i--){
                        dLdYOffset = n*n*(i);
                        dLdZOffset = n*n*(i);
                        dLdYOffset_1 = n*n*(i-1);
                        dLdZOffset_1 = n*n*(i-1);
                        aux_YOffset_1  = n*n*iterNum*d + n*n*(i-1);
                        aux_ZOffset_1  = n*n*iterNum*d + n*n*(i-1);
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                              n,n,n,
                                                                              (T)1,aux_Y + aux_YOffset_1,n,
                                                                              aux_Z + aux_ZOffset_1,n,
                                                                              (T)0,YZ,n);
                        if(error != vl::VLE_Success) {goto done ;}
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::axpy(context,
                                                                             n*n,(T)(-1),
                                                                             I3,ptrdiff_t(1),
                                                                             YZ,ptrdiff_t(1));
                        if(error != vl::VLE_Success) {goto done ;}
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                             n,n,n,
                                                                             (T)1,aux_Z+ aux_ZOffset_1,n,
                                                                             aux_Y + aux_YOffset_1,n,
                                                                             (T)0,ZY,n);
                        if(error != vl::VLE_Success) {goto done ;}
                        Z_dLdZ = iterMemC;Z_dLdZ_Z = iterMemB;ZY_dLdY = iterMemA;
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                             n,n,n,
                                                                             (T)1,aux_Z + aux_ZOffset_1,n,
                                                                             dLdZ + dLdZOffset,n,
                                                                             (T)0,Z_dLdZ,n);
                        if(error != vl::VLE_Success) {goto done ;}
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                             n,n,n,
                                                                             (T)(-0.5),Z_dLdZ,n,
                                                                             aux_Z + aux_ZOffset_1,n,
                                                                             (T)0,Z_dLdZ_Z,n);
                        if(error != vl::VLE_Success) {goto done ;}
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                             n,n,n,
                                                                             (T)(-0.5),ZY,n,
                                                                             dLdY + dLdYOffset,n,
                                                                             (T)0,ZY_dLdY,n);
                        if(error != vl::VLE_Success) {goto done ;}
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                             n,n,n,
                                                                             (T)(-0.5),dLdY + dLdYOffset,n,
                                                                             YZ,n,
                                                                             (T)0,dLdY + dLdYOffset_1,n);
                        if(error != vl::VLE_Success) {goto done ;}
                        matrixAdd_cpu(dLdY + dLdYOffset_1,Z_dLdZ_Z,ZY_dLdY,T(1),T(1),T(1),n*n);
                        Y_dLdY = iterMemC;Y_dLdY_Y = iterMemB;dLdZ_ZY = iterMemA;
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                             n,n,n,
                                                                             (T)1,aux_Y + aux_YOffset_1,n,
                                                                             dLdY + dLdYOffset,n,
                                                                             (T)0,Y_dLdY,n);
                        if(error != vl::VLE_Success) {goto done ;}
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                             n,n,n,
                                                                             (T)(-0.5),Y_dLdY,n,
                                                                             aux_Y + aux_YOffset_1,n,
                                                                             (T)0,Y_dLdY_Y,n);
                        if(error != vl::VLE_Success) {goto done ;}
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                             n,n,n,
                                                                             (T)(-0.5),dLdZ + dLdZOffset,n,
                                                                             ZY,n,
                                                                             (T)0,dLdZ_ZY,n);
                        if(error != vl::VLE_Success) {goto done ;}
                        error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                             'n','n',
                                                                             n,n,n,
                                                                              (T)(-0.5),YZ,n,
                                                                             dLdZ + dLdZOffset,n,
                                                                             (T)0,dLdZ + dLdZOffset_1,n);
                        if(error != vl::VLE_Success) {goto done ;}
                        matrixAdd_cpu(dLdZ + dLdZOffset_1,Y_dLdY_Y,dLdZ_ZY,T(1),T(1),T(1),n*n);
                    }
                    dLdYOffset_1 = n*n*i;
                    dLdZOffset_1 = n*n*i;
                    dLdP  = derData + derDataOffset;
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                          n,n,n,
                                                                         (T)(-0.5),dLdY + dLdYOffset_1,n,
                                                                          P,n,
                                                                         (T)0,dLdP,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                         'n','n',
                                                                         n,n,n,
                                                                         (T)(-0.5),P,n,
                                                                         dLdY + dLdYOffset_1,n,
                                                                         (T)(1),dLdP,n);
                    if(error != vl::VLE_Success) {goto done ;}
                    matrixAdd_cpu(dLdP,dLdY + dLdYOffset_1,dLdZ + dLdZOffset_1,T(1),T(1.5),T(-0.5),n*n);
                    }
            }
            done:
            return context.passError(error, __func__);
        }

    };
} }

template struct vl::impl::cov_sqrtm<vl::VLDT_CPU, float ,vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::cov_sqrtm<vl::VLDT_CPU, double, vl::VLDT_Double> ;
#endif
