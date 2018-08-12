// @file  mpn_cov_gpu.cu
// @brief MPN-COV implementation (GPU)
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2017 Peihua Li and Jiangtao Xie

All rights reserved.
*/

#include "nnmpn_cov_blas.hpp"
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
                                                   T aux_value,
                                                   const ptrdiff_t n)
{
    CUDA_KERNEL_LOOP(index,n){
        a[index*(n+1)]  =  a[index*(n+1)] + aux_value;
    }
}
     

template<typename T> __global__ void processEig_kernel(T* diagS,
                                                  T* diagSroot,
                                                  T* diagSderi,
                                                  T* dev_s,
                                                  ptrdiff_t n,
                                                  int Dmin,
                                                  T epsilon,
                                                  T alpha)
{
    CUDA_KERNEL_LOOP(i,Dmin){
        diagS[i] = dev_s[i];
        diagSroot[i] = pow(dev_s[i] + epsilon,alpha);
        diagSderi[i] = alpha * pow(dev_s[i] + epsilon,alpha - 1);
    }
}
template<typename T> inline void
processEigResults_gpu(T *diagS,
                      T *diagSroot,
                      T *diagSderi,
                      T *dev_s,
                      T *aux_D,
                      ptrdiff_t n,
                      T epsilon,
                      T alpha)
{
    int Dmin = (int)(*aux_D);
    processEig_kernel<T>
        <<<GET_BLOCKS(Dmin),VL_CUDA_NUM_THREADS>>>(diagS,diagSroot,diagSderi,dev_s,n,Dmin,epsilon,alpha);
}
template<typename T> __global__ void getOutput_kernel(T* a,
                                                      T* b,
                                                      ptrdiff_t n)
{
    int lda = n,offset,idx = 0;
    CUDA_KERNEL_LOOP(i,n){
        offset = i;
        for(int j = 0;j < offset + 1;j ++) {
            idx = i * (i + 1) / 2 + j;
            a[idx] = b[i * lda + j];
        }
    }
}
template<typename T>  __host__ void
getOutput_gpu(T* output,
              T* result,
              ptrdiff_t n)
{
    getOutput_kernel<T><<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(output,result,n);
}
template<typename T> __global__ void getdLdCfromderOutput_kernel(T* a,
                                                                 T const* b,
                                                                 ptrdiff_t n)
{
    int lda = n,offset,idx = 0;
    CUDA_KERNEL_LOOP(i,n){
        offset = i;
        for(int j = 0;j < offset + 1;j ++){
            idx = i * (i + 1) / 2 + j;
            a[i * lda + j] = b[idx];
        }
    }
}
template<typename T>  __host__ void
getdLdCfromderOutput_gpu(T* dLdC,
                         T const* derOutput,
                         ptrdiff_t n)
{
    getdLdCfromderOutput_kernel<T><<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(dLdC,derOutput,n);
}

template<typename T> __global__ void symmetric_kernel(T* a,
                                                      int n)
{
    int lda = n,offset;
    CUDA_KERNEL_LOOP(i,n){
        offset = i;
        for(int j = offset;j < n;j ++) {
            a[i * lda + j] = (a[i * lda + j] + a[j * lda + i]) / 2;
            a[j * lda + i] = a[i * lda + j];
        }
    }
}

template<typename T>  __host__ void
symmetric_gpu(T* a,
              ptrdiff_t n)
{
    symmetric_kernel<T>
        <<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(a,n);
}

template<typename T> __global__ void computeKvalueFunction_kernel(T* a,
                                                                  int Dmin,
                                                                  int n)
{
    int lda = n,offset;
    CUDA_KERNEL_LOOP(i, Dmin) {
        offset = i;
        for(int j = offset;j < Dmin;j ++) {
            a[i * lda + j] = 1 / (a[i * lda + j] - a[j * lda + i]);
            a[j * lda + i] = - a[i * lda + j];
        }
    }
}

template<typename T>  __host__ void
computeKvalueFunction_gpu(T* a,
                          ptrdiff_t Dmin,
                          ptrdiff_t n)
{
    computeKvalueFunction_kernel<T>
        <<<GET_BLOCKS(Dmin),VL_CUDA_NUM_THREADS>>>(a,Dmin,n);
}
template<typename T> __global__ void clearInf_kernel(T* a,
                                                     int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(a[i] > 3e38 || a[i] < -3e38) {
        a[i] = 0;
    }
}
template<typename T> __host__ void
clearInf(T* a,
         ptrdiff_t n)
{
    clearInf_kernel<T>
        <<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(a,(int)n);
}
template<typename T> __global__ void computeDvalueFunction_kernel(T* D,
                                                                  T* a,
                                                                  T* b,
                                                                  T* c,
                                                                  int n)
{
    int j,lda = n;
    CUDA_KERNEL_LOOP(i, n) {
        for(j = 0;j < n;j++){
            D[i * lda + j] = - a[i * lda + j] * b[i * lda + j];
        }
        D[i * (n+1)] += c[i * (n+1)];
    }
}
template<typename T>  __host__ void
computeDvalueFunction_gpu(T* D,
                          T* a,
                          T* b,
                          T* c,
                          ptrdiff_t n)
{                                       
    computeDvalueFunction_kernel<T>
        <<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(D,a,b,c,(int)n);
}

namespace vl { namespace impl {
    template<typename T,vl::DataType dataType>
    struct mpn_cov<vl::VLDT_GPU,T,dataType>
    {
        static vl::ErrorCode
            forward(Context& context,
                    T* output,
                    T const* data,
                    T* aux_S,
                    T* aux_V,
                    T* aux_D,
                    size_t height, size_t width, size_t depth, size_t num,
                    T epsilon,
                    T alpha)
        {
            vl::ErrorCode error;
            ptrdiff_t m = height,n = width,L = depth,d;
            ptrdiff_t dataOffset,resultOffset;
            ptrdiff_t aux_VOffset,aux_DOffset;
            ptrdiff_t outputOffset;
            ptrdiff_t pOffset,sOffset;
            ptrdiff_t diagSOffset,diagSrootOffset,diagSderiOffset;
            unsigned int workspaceSize =  (unsigned int)(n*n*(3 + L*1) + 2*n*L + m*n + m*m);
            T* workspace = (T*)context.getWorkspace(vl::VLDT_GPU , workspaceSize*sizeof(T));
            T* S         = workspace;
            T* V         = S + n * L;
            T* diag_S    = V + n * n;
            T* V_diag_S = diag_S + n * L;
            T* result     = V_diag_S + n * n;
            T* II         = result + n * n * L;
            T* cov_work  = II + m*m;
            
            T* P         = aux_V;
            T aux_I_value= -(T)1 / m / m;
            gpuMemset(m*m, aux_I_value, II);
            gpuMemset(2*n*n, T(0), V);
            init_I_kernel<T><<<GET_BLOCKS(m),VL_CUDA_NUM_THREADS>>>(II,(T)1/m,m);
            for(d = 0;d < L; d++){
                dataOffset    = d*m*n;
                pOffset       = d*n*n;
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::cov(context,
                                                                    data + dataOffset,
                                                                    P + pOffset,II,cov_work,
                                                                    m,n);
                symmetric_gpu(P + pOffset,n);
                if(error != VLE_Success) {goto done;}        
            }
            error = vl::impl::blas<vl::VLDT_GPU, dataType>::eig(context, P, S, aux_D, n,L);
            if(error != VLE_Success) {goto done;}
            for(d = 0;d < L; d++){
                diagSOffset     = d*n;
                diagSrootOffset = (d + L) * n;
                diagSderiOffset = (d + 2*L) * n;
                sOffset         = d*n;
                aux_DOffset     = d;
                processEigResults_gpu(aux_S + diagSOffset,aux_S + diagSrootOffset,aux_S + diagSderiOffset,
                                   S + sOffset,aux_D + aux_DOffset, n, epsilon, alpha);
            }
            for(d = 0;d < L; d++){
                aux_VOffset   = d*n*n;
                diagSrootOffset = (d + L) * n;
                resultOffset = d*n*n;
                error = vl::impl::blas<vl::VLDT_GPU , dataType>::dgmm(context,
                                                                      'r',
                                                                      n,n,
                                                                      aux_V + aux_VOffset, n,
                                                                      aux_S + diagSrootOffset, ptrdiff_t(1),
                                                                      V_diag_S, n);

                if(error != VLE_Success) {goto done;}
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
                                                                     'n','t',
                                                                     n,n,n,
                                                                     T(1),V_diag_S,n,
                                                                     aux_V + aux_VOffset,n,
                                                                     T(0),result + resultOffset,n);
                if(error != VLE_Success) {goto done;}    
            }    
            for(d = 0; d < L; d++) {
                resultOffset = d*n*n;
                outputOffset = d*n*(n+1)/2;
                getOutput_gpu(output + outputOffset,result + resultOffset,n);
            }
        done:
            return context.passError(error, __func__);
        }
        static vl::ErrorCode
            backward(Context& context,
                     T* derData,
                     T const* data,
                     T const* derOutput,
                     T const* aux_S,
                     T const* aux_V,
                     T const* aux_D,
                     size_t height, size_t width, size_t depth, size_t num,
                     T epsilon,
                     T alpha)
        {
            vl::ErrorCode error;
            int i;
            ptrdiff_t h = height,w = width, d = depth, L = num;
            ptrdiff_t m = h * w , n = d,Dmin = 0;
            ptrdiff_t dataOffset;
            ptrdiff_t derOutputOffset;
            ptrdiff_t diagSOffset,diagSrootOffset,diagSderiOffset;
            ptrdiff_t aux_V_Offset,aux_DOffset;
            ptrdiff_t derDataOffset;
            unsigned int workspaceSize = (unsigned int)(10*n*n + n + 2*m*n + m*m);
            T* workspace = (T*)context.getWorkspace(vl::VLDT_GPU , workspaceSize*sizeof(T));
            T* dLdC         = workspace; 
            T* dLdC_V    = dLdC + n*n;
            T* dLdV         = dLdC_V  + n*n;
            T* V_diagS   = dLdV  + n*n;
            T* dLdS         = V_diagS + n*n;
            T* K         = dLdS + n*n;
            T* allOnes   = K + n*n;
            T* II         = allOnes + n;
            T* I_X       = II + m*m;
            T* Vt_dLdV   = I_X + m*n;
            T* D         = Vt_dLdV + n*n;
            T* V_D       = D + n*n;
            T* V_D_Vt    = V_D + n*n;
            T aux_I_value= -(T)1 / m / m;
            gpuMemset(m*m, aux_I_value, II);
            gpuMemset(n, (T)1, allOnes);
            init_I_kernel<T><<<GET_BLOCKS(m),VL_CUDA_NUM_THREADS>>>(II,(T)1/m,m);
            for(i = 0;i < L;i++){
                dataOffset      = i * m * n;
                derOutputOffset = i * n * (n + 1) / 2;
                diagSOffset     = i * n;
                diagSrootOffset = (i + L) * n;
                diagSderiOffset = (i + 2*L) * n;
                aux_V_Offset    = i * n * n;
                derDataOffset   = i * m * n;
                aux_DOffset     = i;
                gpuMemset(n*n, (T)0, dLdC); //dLdC
                gpuMemset(n*n, (T)0, D);
                getdLdCfromderOutput_gpu(dLdC, derOutput + derOutputOffset, n);
                symmetric_gpu(dLdC,n);
                Dmin = (ptrdiff_t)(*(aux_D + aux_DOffset));
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     n,n,n,
                                                                     T(2),dLdC,n,
                                                                     aux_V + aux_V_Offset,n,
                                                                     T(0),dLdC_V,n); 
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_GPU , dataType>::dgmm(context,
                                                                      'r',
                                                                      n,n,
                                                                      dLdC_V,n,
                                                                      aux_S + diagSrootOffset,
                                                                      ptrdiff_t(1),
                                                                      dLdV,n);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_GPU , dataType>::dgmm(context,
                                                                      'r',
                                                                      n,n,
                                                                      aux_V + aux_V_Offset,n,
                                                                      aux_S + diagSderiOffset,
                                                                      ptrdiff_t(1),
                                                                      V_diagS,n);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
                                                                     't','n',
                                                                     n,n,n,
                                                                     T(0.5),V_diagS,n,
                                                                     dLdC_V,n,
                                                                     T(0),dLdS, n);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     n,n,(ptrdiff_t)1,
                                                                     T(1),aux_S + diagSOffset,n,
                                                                     allOnes,(ptrdiff_t)1,
                                                                     T(0),K,n);  
                                                                     
                if(error != vl::VLE_Success) {goto done ;}
                computeKvalueFunction_gpu<T>(K,Dmin,n);
                clearInf<T>(K,n*n);
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     m,n,m,
                                                                     T(1),II,m,
                                                                     data + dataOffset,m,
                                                                     T(0),I_X,m);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
                                                                     't','n',
                                                                     n,n,n,
                                                                     T(1),aux_V + aux_V_Offset,n,
                                                                     dLdV,n,
                                                                     T(0),Vt_dLdV,n);
                if(error != vl::VLE_Success) {goto done ;}
                computeDvalueFunction_gpu<T>(D,Vt_dLdV,K,dLdS,n);
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     n,n,n,
                                                                     T(1),aux_V + aux_V_Offset,n,
                                                                     D,n,
                                                                     T(0),V_D,n);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
                                                                     'n','t',
                                                                     n,n,n,
                                                                     T(1),V_D,n,
                                                                     aux_V + aux_V_Offset,n,
                                                                     T(0),V_D_Vt,n);
                symmetric_gpu<T>(V_D_Vt,n);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     m,n,n,
                                                                     T(2),I_X,m,
                                                                     V_D_Vt,n,
                                                                     T(0),derData + derDataOffset,m);
                if(error != vl::VLE_Success) {goto done ;}
        }
        done:
            return context.passError(error, __func__);
    }

    };
} }
template struct vl::impl::mpn_cov<vl::VLDT_GPU, float,vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::mpn_cov<vl::VLDT_GPU, double, vl::VLDT_Double> ;
#endif
