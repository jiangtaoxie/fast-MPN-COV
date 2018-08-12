// @file  cov_traceNorm_gpu.cu
// @brief fast MPN-COV implementation (GPU)
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
*/

#include "nncov_traceNorm_blas.hpp"
#include "../data.hpp"
#include <math.h>
#include <memory>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <cassert>
#include "blashelper_gpu.hpp"


#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)   // We import this Macro function  from our Caffe Implementation 

inline int
GET_BLOCKS(const int N)
{
    return (N + VL_CUDA_NUM_THREADS - 1) / VL_CUDA_NUM_THREADS; // We import this function  from our Caffe Implementation 
}

template<typename T> __global__ void set_kernel(const ptrdiff_t n, const T alpha, T* y) 
{
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = alpha;
    }
}

template<typename T> void gpuMemset(const ptrdiff_t n, const T alpha, T* y)
{
    if(alpha == 0){
        cudaMemset(y, 0, sizeof(T)*n);
    }
    set_kernel<T><<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(n , alpha, y);
}

template<typename T> __global__ void init_I_kernel(T* a,
                                                   T  alpha,
                                                   const ptrdiff_t n)
{
    CUDA_KERNEL_LOOP(index,n){
        a[index*(n+1)]  =  alpha;
    }
}




template<typename T> __global__ void traceNormBackward_kernel(T* a,
                                                              T const* alpha,
                                                              T* beta,
                                                              T const* sigma,
                                                              ptrdiff_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        a[i*(n+1)] = a[i*(n+1)] - beta[0]/(alpha[0] * alpha[0]) + sigma[0]/(2.0f * sqrt(alpha[0]));
    }
}

template<typename T>  __host__ void
traceNormBackward_gpu(T* a,
                      T const* alpha,
                      T* beta,
                      T const* sigma,
                      ptrdiff_t n)
{
    traceNormBackward_kernel<T>
        <<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(a,alpha,beta,sigma,n);
}

template<typename T> __global__ void copy_kernel(T* a,
                                                 T* b,
                                                 ptrdiff_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] = b[i];
    }
}

template<typename T> __global__ void scale_kernel(T* a,
                                                  T const* b,
                                                  T const* alpha,
                                                  ptrdiff_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] = (1.0f / alpha[0]) * b[i];
    }
}
template<typename T> __global__ void scale2_kernel(T* a,
                                                   T const* b,
                                                   T const* alpha,
                                                   ptrdiff_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] = sqrt(alpha[0]) * b[i];
    }
}
template<typename T> __global__ void mul_kernel(T* a,
                                                T const* alpha,
                                                ptrdiff_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] = a[i] / ( 2.0f * alpha[0]);
    }
}
namespace vl { namespace impl {
    template<typename T,vl::DataType dataType>
    struct cov_traceNorm<vl::VLDT_GPU,T,dataType>
    {
        static vl::ErrorCode
            forward(Context& context,
                    T* output,
                    T const* data,
                    T* aux_T,
                    size_t height, size_t width, size_t depth, size_t num)
        {
            vl::ErrorCode error;
            ptrdiff_t m = height,n = width,L = num,d,i;
            ptrdiff_t dataOffset;
            ptrdiff_t aux_TOffset;
            ptrdiff_t outputOffset;
            unsigned int workspaceSize =  (unsigned int)(n*n);
            T* workspace = (T*)context.getWorkspace(vl::VLDT_GPU , workspaceSize*sizeof(T));
            T* I1        = workspace;
            gpuMemset(n*n, T(0), I1);
            init_I_kernel<T><<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(I1,(T)1,n);
            for(d = 0;d < L; d++){ // Trace Norm
                aux_TOffset  = d;
                outputOffset = d*n*n;
                dataOffset   = d*n*n;
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::dot(context,
                                                                    n*n,
                                                                    data + dataOffset,ptrdiff_t(1),
                                                                    I1,ptrdiff_t(1),
                                                                    aux_T + aux_TOffset);
                if(error != vl::VLE_Success) {goto done ;}
            }
            for(d = 0;d < L; d++){
                aux_TOffset  = d;
                outputOffset = d*n*n;
                dataOffset   = d*n*n;
                scale_kernel<T><<<GET_BLOCKS(n*n),VL_CUDA_NUM_THREADS>>>(output + outputOffset,data + dataOffset,aux_T + aux_TOffset,n*n);
            }
            
        done:
            return context.passError(error, __func__);
        }
        static vl::ErrorCode
            backward(Context& context,
                     T* derData,
                     T const* data,
                     T const* derOutput,
                     T const* derOutput_aux,
                     T const* aux_T,
                     size_t height, size_t width, size_t depth, size_t num)
        {
            vl::ErrorCode error;
            ptrdiff_t m = height,n = width,L = num,d;
            ptrdiff_t derDataOffset,aux_TOffset,derOutputOffset;
            ptrdiff_t dataOffset,derOutput_auxOffset,P_dot_dLdPOffset;
            unsigned int workspaceSize =  (unsigned int)(L);
            T* workspace = (T*)context.getWorkspace(vl::VLDT_GPU , workspaceSize*sizeof(T));
            T* P_dot_dLdP = workspace;
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
            for(d = 0;d < L;d++){
                dataOffset    = d*m*n;
                derDataOffset = d*m*n;
                derOutputOffset = d*m*n;
                derOutput_auxOffset = d;
                aux_TOffset   = d;
                P_dot_dLdPOffset = d;
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::dot(context,
                                                                    n*n,
                                                                    data + dataOffset,ptrdiff_t(1),
                                                                    derOutput + derOutputOffset,ptrdiff_t(1),
                                                                    P_dot_dLdP + P_dot_dLdPOffset);
                if(error != vl::VLE_Success) {goto done ;}
            }
            for(d = 0;d < L; d++){
                dataOffset    = d*m*n;
                derDataOffset = d*m*n;
                derOutputOffset = d*m*n;
                derOutput_auxOffset = d;
                aux_TOffset   = d;
                P_dot_dLdPOffset = d;
                scale_kernel<T><<<GET_BLOCKS(n*n),VL_CUDA_NUM_THREADS>>>(derData + derDataOffset,derOutput + derOutputOffset,aux_T + aux_TOffset,n*n);
                traceNormBackward_gpu(derData + derDataOffset,aux_T + aux_TOffset,P_dot_dLdP + P_dot_dLdPOffset,derOutput_aux + derOutput_auxOffset, n);
            }

            done:
            return context.passError(error, __func__);
           
        }
        static vl::ErrorCode
            forward_aux(Context& context,
                        T* output,
                        T const* data,
                        T* aux_T,
                        size_t height, size_t width, size_t depth, size_t num)
        {
            vl::ErrorCode error;
            ptrdiff_t length = depth,L = num,d;
            ptrdiff_t outputOffset,aux_TOffset,dataOffset;
            for(d = 0; d < L; d++){
                outputOffset = d*length;
                dataOffset = d*length;
                aux_TOffset  = d;
                /*error = vl::impl::blas<vl::VLDT_CPU, dataType>::scal(context,
                                                                     length,
                                                                     alpha,
                                                                     output + OutputOffset,ptrdiff_t(1));
                if(error != vl::VLE_Success) {goto done ;}*/
                scale2_kernel<T><<<GET_BLOCKS(length),VL_CUDA_NUM_THREADS>>>(output + outputOffset,data + dataOffset,aux_T + aux_TOffset,length);
                error = vl::VLE_Success;
            }
            done:
            return context.passError(error, __func__);
        }
        static vl::ErrorCode
            backward_aux(Context& context,
                         T* derData,
                         T* derData_aux,
                         T const* data,
                         T const* derOutput,
                         T const* aux_T,
                         size_t height, size_t width, size_t depth, size_t num)
        {
            vl::ErrorCode error;
            ptrdiff_t length = depth,L = num,d;
            ptrdiff_t dataOffset,aux_TOffset;
            ptrdiff_t derDataOffset,derData_auxOffset;
            ptrdiff_t derOutputOffset;
            for(d = 0; d < L; d++){
                derDataOffset   = d*length;
                derOutputOffset = d*length;
                derData_auxOffset = d;
                dataOffset = d*length;
                aux_TOffset= d;
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::dot(context,
                                                                    length,
                                                                    data + dataOffset,ptrdiff_t(1),
                                                                    derOutput + derOutputOffset,ptrdiff_t(1),
                                                                    derData_aux + derData_auxOffset);
                if(error != vl::VLE_Success) {goto done ;}
                
            }
            for(d = 0; d < L; d++){
                derDataOffset   = d*length;
                derOutputOffset = d*length;
                derData_auxOffset = d;
                dataOffset = d*length;
                aux_TOffset= d;
                scale2_kernel<T><<<GET_BLOCKS(length),VL_CUDA_NUM_THREADS>>>(derData + derDataOffset,derOutput + derOutputOffset,aux_T + aux_TOffset,length);
            }
            done:
            return context.passError(error, __func__);
        }


    };
} }
template struct vl::impl::cov_traceNorm<vl::VLDT_GPU, float,vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::cov_traceNorm<vl::VLDT_GPU, double, vl::VLDT_Double> ;
#endif
