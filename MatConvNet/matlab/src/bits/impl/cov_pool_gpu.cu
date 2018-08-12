// @file  cov_pool_gpu.cu
// @brief fast MPN-COV implementation (GPU)
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
template<typename T> __global__ void init_II_kernel(T* a,
                                                   T aux_value,
                                                   const ptrdiff_t n)
{
    CUDA_KERNEL_LOOP(index,n){
        a[index*(n+1)]  =  a[index*(n+1)] + aux_value;
    }
}


template<typename T> __global__ void symmetric_kernel(T const* a,
													  T* b,
                                                      int n)
{
    int lda = n,offset;
    CUDA_KERNEL_LOOP(i,n){
        offset = i;
        for(int j = offset;j < n;j ++) {
			b[i * lda + j] = (a[i * lda + j] + a[j * lda + i]) / 2.0f;
			b[j * lda + i] = b[i * lda + j];
        }
    }
}

template<typename T>  __host__ void
symmetric_gpu(T const* a,
              T* b,
              ptrdiff_t n)
{
    symmetric_kernel<T>
        <<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(a,b,n);
}


namespace vl { namespace impl {
    template<typename T,vl::DataType dataType>
    struct cov_pool<vl::VLDT_GPU,T,dataType>
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
			T* workspace = (T*)context.getWorkspace(vl::VLDT_GPU , workspaceSize*sizeof(T));
			T* II        = workspace;
            T* cov_work  = II + m*m;
			T aux_I_value= -(T)1 / m / m;
            gpuMemset(m*m, aux_I_value, II);
			init_II_kernel<T><<<GET_BLOCKS(m),VL_CUDA_NUM_THREADS>>>(II,(T)1/m,m);
			for(d = 0;d < L; d++){ //Covariance
                dataOffset    = d*m*n;
				outputOffset  = d*n*n;
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::cov(context,
                                                                    data + dataOffset,
																	output + outputOffset,II,cov_work,
                                                                    m,n);
                symmetric_gpu(output + outputOffset,output + outputOffset,n);
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
			T* workspace = (T*)context.getWorkspace(vl::VLDT_GPU , workspaceSize*sizeof(T));
			T* I_X       = workspace;
			T* II        = I_X + m*n;
			T* dLdP      = II + m*m;
			T aux_I_value= -(T)1 / m / m;
			gpuMemset(m*m, aux_I_value, II);
			init_II_kernel<T><<<GET_BLOCKS(m),VL_CUDA_NUM_THREADS>>>(II,(T)1/m,m);

			for(d = 0;d < L;d++){
				dataOffset      = d*m*n;
				derOutputOffset = d*n*n;
				derDataOffset   = d*m*n;
				symmetric_gpu(derOutput + derOutputOffset,dLdP,n);
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     m,n,m,
                                                                     T(1),II,m,
                                                                     data + dataOffset,m,
                                                                     T(0),I_X,m);
				if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
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
template struct vl::impl::cov_pool<vl::VLDT_GPU, float,vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::cov_pool<vl::VLDT_GPU, double, vl::VLDT_Double> ;
#endif
