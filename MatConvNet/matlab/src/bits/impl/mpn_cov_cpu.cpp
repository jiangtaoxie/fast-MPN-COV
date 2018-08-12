// @file  mpn_cov_gpu.cpp
// @brief MPN-COV implementation (CPU)
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
#include "blashelper_cpu.hpp"


template<typename T> inline void
processEigResults_cpu(T *diagS,
                      T *diagSroot,
                      T *diagSderi,
                      T *S_diag,
                      T *host_s,
                      T *aux_D,
                      ptrdiff_t n,
                      T epsilon,
                      T alpha)
{
    int i;
    int Dmin = (int)(*aux_D);
    for(i = 0; i < Dmin; i ++){
        diagS[i] = host_s[i];
        diagSroot[i] = pow(host_s[i] + epsilon, alpha);
        diagSderi[i] = alpha * pow(host_s[i] + epsilon,alpha - 1);
        S_diag[i*(n+1)] = diagSroot[i];
    }
}


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
            a[i * lda + j] = (a[i * lda + j] + a[j * lda + i]) / 2;
            a[j * lda + i] = a[i * lda + j];
        }
    }
}

template<typename T> inline void
diagFunction_cpu(T* S,
                 T const* aux_S,
                 ptrdiff_t n)
{
    int i;
    for(i = 0;i < n;i ++){
        S[i*(n+1)] = aux_S[i];
    }
}

template<typename T> inline void
computeKvalueFunction_cpu(T* a,
                      ptrdiff_t n,
                      ptrdiff_t Dmin)
{
    int lda = (int)n;
    int offset = 0;
    int i,j;
    for(i = 0;i < Dmin;i ++){
        offset = i;
        for(j = offset;j < Dmin;j ++) {
            a[i * lda + j] = 1 / (a[i * lda + j] - a[j * lda + i]);
            a[j * lda + i] = - a[i * lda + j];
        }
    }
}
template<typename T> inline void
clearInf_cpu(T* a,
			 ptrdiff_t n)
{
	int i;
	for(i = 0;i < n;i++) {
		if(a[i] > 3e38 || a[i] < -3e38) {
        a[i] = 0;
        }
	}
}
template<typename T> inline void
computeDvalueFunction_cpu(T* D,
                      T* a,
                      T* b,
                      T* c,
                      ptrdiff_t n)
{
    int i,j,lda = int(n);
    for(i = 0;i < n;i ++){
        for(j = 0;j < n;j ++) {
            D[i * lda + j] = - a[i * lda + j] * b[i * lda + j]; // D = a * b' ;
        }
        D[i*(n+1)] += c[i*(n+1)];
    }
    
}
namespace vl { namespace impl {
    template<typename T,vl::DataType dataType>
    struct mpn_cov<vl::VLDT_CPU,T,dataType>
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
            ptrdiff_t pOffset,sOffset,sdiagOffset;
            ptrdiff_t diagSOffset,diagSrootOffset,diagSderiOffset;
            unsigned int workspaceSize =  (unsigned int)(n*n*(1 + L*2) + 2*n*L + m*n + m*m);
            T* workspace = (T*)context.getWorkspace(vl::VLDT_CPU , workspaceSize*sizeof(T));
            T* S         = workspace;
            T* S_diag    = S + n * L;
            T* diag_S_Vt = S_diag + n * n * L;
            T* result     = diag_S_Vt + n * n;
            T* II         = result + n * n * L;
            T* cov_work  = II + m*m;
            
            T* P         = aux_V;
            T aux_I_value= -(T)1 / m / m;
            vl::impl::operations<vl::VLDT_CPU, T>::fill(II, m*m, aux_I_value);
	    vl::impl::operations<vl::VLDT_CPU, T>::fill(S_diag, n*n*L, T(0));
            for(d = 0;d < m;d++){
                II[d*(m+1)] =  II[d*(m+1)] + (T)1 / m;    // init I
            }
            for(d = 0;d < L; d++){
                dataOffset    = d*m*n;
                pOffset       = d*n*n;
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::cov(context,
                                                                    data + dataOffset,
                                                                    P + pOffset,II,cov_work,
                                                                    m,n);
                symmetric_cpu(P + pOffset,n);
                if(error != VLE_Success) {goto done;}        
            }
            error = vl::impl::blas<vl::VLDT_CPU, dataType>::eig(context, P, S, aux_D, n,L);
            if(error != VLE_Success) {goto done;}
            for(d = 0;d < L; d++){
                diagSOffset     = d*n;
                diagSrootOffset = (d + L) * n;
                diagSderiOffset = (d + 2*L) * n;
                sOffset         = d*n;
                aux_DOffset     = d;
                sdiagOffset     = d*n*n;
                processEigResults_cpu(aux_S + diagSOffset,aux_S + diagSrootOffset,aux_S + diagSderiOffset,
                                   S_diag + sdiagOffset,S + sOffset,aux_D + aux_DOffset, n, epsilon, alpha);
            }
            for(d = 0;d < L; d++){
                aux_VOffset   = d*n*n;
                sdiagOffset     = d*n*n;
                resultOffset = d*n*n;
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                     'n','t',
                                                                      n,n,n,
                                                                      T(1),S_diag + sdiagOffset,n,
                                                                      aux_V + aux_VOffset,n,
                                                                      T(0),diag_S_Vt,n);

               if(error != VLE_Success) {goto done;}
               error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                    'n','n',
                                                                     n,n,n,
                                                                     T(1),aux_V + aux_VOffset,n,
                                                                     diag_S_Vt,n,
                                                                     T(0),result + resultOffset,n);
               if(error != VLE_Success) {goto done;}
            }    
            for(d = 0; d < L; d++) {
                resultOffset = d*n*n;
                outputOffset = d*n*(n+1)/2;
                getOutput_cpu(output + outputOffset,result + resultOffset,n);
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
            ptrdiff_t m = h * w , n = d,Dmin;
            ptrdiff_t dataOffset;
            ptrdiff_t derOutputOffset;
            ptrdiff_t aux_V_Offset,aux_D_Offset;
            ptrdiff_t derDataOffset;
            ptrdiff_t diagSOffset,diagSrootOffset,diagSderiOffset;
            unsigned int workspaceSize = (unsigned int)(11*n*n + 2*n + 2*m*n+m*m);
            T* workspace = (T*)context.getWorkspace(vl::VLDT_CPU , workspaceSize*sizeof(T));
            T* dLdC         = workspace;
            T* S         = dLdC + n*n; 
            T* dLdC_V    = S + n*n;
            T* dLdV         = dLdC_V  + n*n;
            T* diag_S_Vt = dLdV  + n*n;
            T* dLdS         = diag_S_Vt + n*n;
            T* K         = dLdS + n*n;
            T* diag_S    = K + n*n;
            T* allOnes   = diag_S + n;
            T* II         = allOnes + n;
            T* I_X       = II + m*m;
            T* Vt_dLdV   = I_X + m*n;
            T* D         = Vt_dLdV + n*n;
            T* V_D       = D + n*n;
            T* V_D_Vt    = V_D + n*n;
            T aux_I_value= -(T)1 / m / m;
            vl::impl::operations<vl::VLDT_CPU, T>::fill(II, m*m, aux_I_value);
            vl::impl::operations<vl::VLDT_CPU, T>::fill(S, n*n , (T)0);
            vl::impl::operations<vl::VLDT_CPU, T>::fill(allOnes, n, (T)1);
            
            for(i = 0;i < m;i++){
                II[i*(m+1)] =  II[i*(m+1)] + (T)1 / m;    // init I
            }
            for(i = 0;i < L;i++){
                dataOffset      = i * m * n;
                derOutputOffset = i * n * (n + 1) / 2;
                aux_V_Offset    = i * n * n;
                derDataOffset   = i * m * n;
                diagSOffset     = i * n;
                diagSrootOffset = (i + L) * n;
                diagSderiOffset = (i + 2*L) * n;
                aux_D_Offset    = i;
                vl::impl::operations<vl::VLDT_CPU, T>::fill(dLdC, n*n , (T)0);
                getdLdCfromderOutput_cpu(dLdC, derOutput + derOutputOffset, n);
                symmetric_cpu(dLdC,n);
                diagFunction_cpu(S,aux_S + diagSrootOffset,n);
                memcpy(diag_S,aux_S + diagSOffset,sizeof(T)*n);
                Dmin = (ptrdiff_t)(*(aux_D + aux_D_Offset));
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     n,n,n,
                                                                     T(1),dLdC,n,
                                                                     aux_V + aux_V_Offset,n,
                                                                     T(0),dLdC_V,n);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                      'n','n',
                                                                      n,n,n,
                                                                      T(2),dLdC_V,n,
                                                                      S,n,
                                                                      T(0),dLdV,n);
                if(error != vl::VLE_Success) {goto done ;}
                diagFunction_cpu(S,aux_S + diagSderiOffset,n);
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                     'n','t',
                                                                     n,n,n,
                                                                     T(1),S,n,
                                                                     aux_V + aux_V_Offset,n,
                                                                     T(0),diag_S_Vt,n);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     n,n,n,
                                                                     T(1),diag_S_Vt,n,
                                                                     dLdC_V,n,
                                                                     T(0),dLdS, n);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     n,n,(ptrdiff_t)1,
                                                                     T(1),diag_S,n,
                                                                     allOnes,(ptrdiff_t)1,
                                                                     T(0),K,n);
                if(error != vl::VLE_Success) {goto done ;}
                computeKvalueFunction_cpu(K,n,Dmin);
				clearInf_cpu(K,n*n);
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     m,n,m,
                                                                     T(1),II,m,
                                                                     data + dataOffset,m,
                                                                     T(0),I_X,m);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                     't','n',
                                                                     n,n,n,
                                                                     T(1),aux_V + aux_V_Offset,n,
                                                                     dLdV,n,
                                                                     T(0),Vt_dLdV,n);
                computeDvalueFunction_cpu(D,Vt_dLdV,K,dLdS,n);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     n,n,n,
                                                                     T(1),aux_V + aux_V_Offset,n,
                                                                     D,n,
                                                                     T(0),V_D,n);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
                                                                     'n','t',
                                                                     n,n,n,
                                                                     T(1),V_D,n,
                                                                     aux_V + aux_V_Offset,n,
                                                                     T(0),V_D_Vt,n);
                symmetric_cpu(V_D_Vt,n);
                if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::gemm(context,
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

template struct vl::impl::mpn_cov<vl::VLDT_CPU, float ,vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::mpn_cov<vl::VLDT_CPU, double, vl::VLDT_Double> ;
#endif
