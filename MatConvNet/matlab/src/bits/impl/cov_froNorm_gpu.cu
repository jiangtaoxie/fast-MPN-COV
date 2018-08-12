// @file  cov_froNorm_gpu.cu
// @brief fast MPN-COV implementation (GPU)
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
*/

#include "nncov_froNorm_blas.hpp"
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




template<typename T> __global__ void froNormBackward_kernel(T* a,
                                                            T const* b,
					                                        T const* alpha,
				                                            ptrdiff_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        a[i] += alpha[0] * b[i];
    }
}

template<typename T>  __host__ void
froNormBackward_gpu(T* a,
                    T const* b,
					T const* alpha,
				    ptrdiff_t n)
{
    froNormBackward_kernel<T>
		<<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(a,b,alpha,n);
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
template<typename T> __global__ void sqrt_kernel(T* a,
												 ptrdiff_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		a[i] = sqrt(a[i]);
	}
}
template<typename T> __global__ void computeDerdataAux_kernel(T* a,
															  T const* b,
															  ptrdiff_t n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		a[i] /= 2 * pow(b[i], (T)3.0f /2.0f);
	}
}
template<typename T> __global__ void computeGradientCoef_kernel(T* a,
																T const* b,
																T const* c,
																T const* d,
												                ptrdiff_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		a[i] = b[i] - c[i] / ( pow(d[i], (T)3.0f));
	}
}
namespace vl { namespace impl {
    template<typename T,vl::DataType dataType>
    struct cov_froNorm<vl::VLDT_GPU,T,dataType>
    {
        static vl::ErrorCode
            forward(Context& context,
                    T* output,
                    T const* data,
					T* aux_T,
                    size_t height, size_t width, size_t depth, size_t num)
        {
            vl::ErrorCode error;
            ptrdiff_t m = height,n = width,L = num,d;
			ptrdiff_t dataOffset;
			ptrdiff_t aux_TOffset;
			ptrdiff_t outputOffset;
			for(d = 0;d < L; d++){ // Fro-Norm
				aux_TOffset = d;
				dataOffset  = d*n*n;
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::dot(context,
					                                                n*n,
																	data + dataOffset,ptrdiff_t(1),
															        data + dataOffset,ptrdiff_t(1),
															        aux_T + aux_TOffset);
				if(error != vl::VLE_Success) {goto done ;}
			}
			sqrt_kernel<T><<<GET_BLOCKS(L),VL_CUDA_NUM_THREADS>>>(aux_T,L);
			for(d = 0;d < L; d++){
				aux_TOffset = d;
				outputOffset = d*n*n;
				dataOffset  = d*n*n;
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
			ptrdiff_t derDataOffset,aux_TOffset,gradientValueOffset;
			ptrdiff_t dataOffset,derOutput_auxOffset,P_dot_dLdPOffset;
			ptrdiff_t derOutputOffset;
			unsigned int workspaceSize =  (unsigned int)(2*L);
			T* workspace = (T*)context.getWorkspace(vl::VLDT_GPU , workspaceSize*sizeof(T));
			T* P_dot_dLdP = workspace;
			T* gradientValue = P_dot_dLdP + L;
			for(d = 0;d < L;d++){
				dataOffset    = d*m*n;	
				derOutput_auxOffset = d;
				derOutputOffset = d*m*n;
				aux_TOffset   = d;
				P_dot_dLdPOffset = d;
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::dot(context,
					                                                n*n,
															        data + dataOffset,ptrdiff_t(1),
																	derOutput + derOutputOffset,ptrdiff_t(1),
															        P_dot_dLdP + P_dot_dLdPOffset);
				if(error != vl::VLE_Success) {goto done ;}
			}
			for(d = 0;d < L;d++){
				aux_TOffset   = d;
				derDataOffset = d*m*n;
				derOutputOffset = d*m*n;
				scale_kernel<T><<<GET_BLOCKS(n*n),VL_CUDA_NUM_THREADS>>>(derData + derDataOffset,derOutput + derOutputOffset,aux_T + aux_TOffset,n*n);
			}
			computeGradientCoef_kernel<T><<<GET_BLOCKS(L),VL_CUDA_NUM_THREADS>>>(gradientValue,derOutput_aux,P_dot_dLdP,aux_T,L);
			for(d = 0;d < L;d++){
				dataOffset    = d*m*n;
				derDataOffset = d*m*n;
				gradientValueOffset = d;
				froNormBackward_gpu(derData + derDataOffset,data + dataOffset,gradientValue + gradientValueOffset, m*n);
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
			computeDerdataAux_kernel<T><<<GET_BLOCKS(L),VL_CUDA_NUM_THREADS>>>(derData_aux,aux_T,L);
			for(d = 0;d < L; d++){
				derDataOffset   = d*length;
				derOutputOffset = d*length;
				aux_TOffset= d;
				scale2_kernel<T><<<GET_BLOCKS(length),VL_CUDA_NUM_THREADS>>>(derData + derDataOffset,derOutput + derOutputOffset,aux_T + aux_TOffset,length);
			}
			done:
            return context.passError(error, __func__);
		}


    };
} }
template struct vl::impl::cov_froNorm<vl::VLDT_GPU, float,vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::cov_froNorm<vl::VLDT_GPU, double, vl::VLDT_Double> ;
#endif
