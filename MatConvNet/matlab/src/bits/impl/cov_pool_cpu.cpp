// @file  cov_pool_gpu.cpp
// @brief fast MPN-COV implementation (CPU)
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
*/
#include "nncov_pool_blas.hpp"
#include "../data.hpp"
#include <math.h>
#include <memory>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <cassert>
#include "blashelper_cpu.hpp"





template<typename T> inline void
symmetric_cpu(T const* a,
              T* b,
              ptrdiff_t n)
{
    int lda = (int)n;
    int offset = 0;
    int i,j;
    for(i = 0;i < n;i ++) {
        offset = i;
        for(j = offset;j < n;j ++) {
            b[i * lda + j] = (a[i * lda + j] + a[j * lda + i]) / 2.0f;
            b[j * lda + i] = b[i * lda + j];
        }
    }
}

namespace vl { namespace impl {
    template<typename T,vl::DataType dataType>
    struct cov_pool<vl::VLDT_CPU,T,dataType>
    {
        static vl::ErrorCode
            forward(Context& context,
                    T* output,
                    T const* data,
                    size_t height, size_t width, size_t depth, size_t num)
        {
            vl::ErrorCode error;
            ptrdiff_t m = height,n = width,L = depth,d;
            ptrdiff_t dataOffset;
            ptrdiff_t outputOffset;
            unsigned int workspaceSize =  (unsigned int)(m*m + m*n);
            T* workspace = (T*)context.getWorkspace(vl::VLDT_CPU , workspaceSize*sizeof(T));
            T* II        = workspace;
            T* cov_work  = II + m*m;
            T aux_I_value= -(T)1 / m / m;
            vl::impl::operations<vl::VLDT_CPU, T>::fill(II, m*m, aux_I_value);
            for(d = 0;d < m;d++){
                II[d*(m+1)] =  II[d*(m+1)] + (T)1 / m;    // init I
            }
            for(d = 0;d < L; d++){ //Covariance
                dataOffset    = d*m*n;
                outputOffset  = d*n*n;
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::cov(context,
                                                                    data + dataOffset,
                                                                    output + outputOffset,II,cov_work,
                                                                    m,n);
                symmetric_cpu(output + outputOffset,output + outputOffset,n);
                if(error != VLE_Success) {goto done;}        
            }
        done:
            return context.passError(error, __func__);
        }
        static vl::ErrorCode
            backward(Context& context,
                     T* derData,
                     T const* data,
                     T const* derOutput,
                     size_t height, size_t width, size_t depth, size_t num)
        {
            vl::ErrorCode error;
            ptrdiff_t m = height*width,n = depth,L = num,d;
            ptrdiff_t derOutputOffset,dataOffset;
            ptrdiff_t derDataOffset;
            unsigned int workspaceSize =  (unsigned int)(m*n + m*m + n*n);
            T* workspace = (T*)context.getWorkspace(vl::VLDT_CPU , workspaceSize*sizeof(T));
            T* I_X       = workspace;
            T* II        = I_X + m*n;
            T* dLdP      = II + m*m;
            T aux_I_value= -(T)1 / m / m;
            vl::impl::operations<vl::VLDT_CPU, T>::fill(II, m*m, aux_I_value);
            /*debug
            mxArray *x;
            mxClassID classID = mxSINGLE_CLASS ;
            mwSize dim[2];
            dim[0] = n;dim[1] = n;
            x  = mxCreateNumericArray(2,dim,classID,mxREAL);
             T *test = (T*)mxGetData(x);
            memcpy(test,YZ,sizeof(T)*n*n);
            mexCallMATLAB(0,NULL,1,&x,"Watch");
            */
            for(d = 0;d < m;d++){
                II[d*(m+1)] =  II[d*(m+1)] + (T)1 / m;    // init I
            }
            for(d = 0;d < L;d++){
                dataOffset      = d*m*n;
                derOutputOffset = d*n*n;
                derDataOffset   = d*m*n;
                symmetric_cpu(derOutput + derOutputOffset,dLdP,n);
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     m,n,m,
                                                                     T(1),II,m,
                                                                     data + dataOffset,m,
                                                                     T(0),I_X,m);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     m,n,n,
                                                                     T(2),I_X,m,
                                                                     dLdP,n,
                                                                     T(0),derData + derDataOffset,m);
                if(error != vl::VLE_Success) {goto done ;}

            }
            done:
            return context.passError(error, __func__);
        }

    };
} }

template struct vl::impl::cov_pool<vl::VLDT_CPU, float ,vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::cov_pool<vl::VLDT_CPU, double, vl::VLDT_Double> ;
#endif
