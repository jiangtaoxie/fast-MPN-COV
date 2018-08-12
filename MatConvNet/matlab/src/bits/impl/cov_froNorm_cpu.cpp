// @file  cov_froNorm_gpu.cpp
// @brief fast MPN-COV implementation (CPU)
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
#include "blashelper_cpu.hpp"







template<typename T> inline void
froNormBackward_cpu(T* a,
                    T const* b,
                    T* alpha,
                    ptrdiff_t n)
{
    int i;
    for(i = 0;i < n;i++){
        a[i] += alpha[0] * b[i];
    }
}
namespace vl { namespace impl {
    template<typename T,vl::DataType dataType>
    struct cov_froNorm<vl::VLDT_CPU,T,dataType>
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
            memcpy(output,data,sizeof(T)*m*n*L);
            for(d = 0;d < L; d++){ // Fro-Norm
                aux_TOffset = d;
                dataOffset  = d*n*n;
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::dot(context,
                                                                    n*n,
                                                                    data + dataOffset,ptrdiff_t(1),
                                                                    data + dataOffset,ptrdiff_t(1),
                                                                    aux_T + aux_TOffset);
                if(error != vl::VLE_Success) {goto done ;}
            }
            for(d = 0;d < L; d++){
                aux_TOffset = d;
                aux_T[aux_TOffset] = sqrt(aux_T[aux_TOffset]);
            }
            for(d = 0;d < L; d++){
                aux_TOffset = d;
                outputOffset = d*n*n;
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::scal(context,
                                                                     n*n,
                                                                    (T)1.0f/aux_T[aux_TOffset],
                                                                    output + outputOffset,ptrdiff_t(1));
                if(error != vl::VLE_Success) {goto done ;}
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
            unsigned int workspaceSize =  (unsigned int)(2*L);
            T* workspace = (T*)context.getWorkspace(vl::VLDT_CPU , workspaceSize*sizeof(T));
            T* P_dot_dLdP = workspace;
            T* gradientValue = P_dot_dLdP + L;
            memcpy(derData,derOutput,sizeof(T)*m*n*L);
            for(d = 0;d < L;d++){
                dataOffset    = d*m*n;
                derDataOffset = d*m*n;        
                derOutput_auxOffset = d;
                aux_TOffset   = d;
                P_dot_dLdPOffset = d;
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::dot(context,
                                                                    n*n,
                                                                    data + dataOffset,ptrdiff_t(1),
                                                                    derData + derDataOffset,ptrdiff_t(1),
                                                                    P_dot_dLdP + P_dot_dLdPOffset);
                if(error != vl::VLE_Success) {goto done ;}
            }
            for(d = 0;d < L;d++){
                aux_TOffset   = d;
                derDataOffset = d*m*n;
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::scal(context,
                                                                     n*n,
                                                                    (T)1.0f/aux_T[aux_TOffset],
                                                                     derData + derDataOffset,ptrdiff_t(1));
                if(error != vl::VLE_Success) {goto done ;}
            }
            for(d = 0;d < L;d++){
                gradientValueOffset = d;
                derOutput_auxOffset = d;
                aux_TOffset   = d;
                P_dot_dLdPOffset = d;
                gradientValue[gradientValueOffset] = derOutput_aux[derOutput_auxOffset] - P_dot_dLdP[P_dot_dLdPOffset] / (pow(aux_T[aux_TOffset],(T)3.0f));
            }
            for(d = 0;d < L;d++){
                dataOffset    = d*m*n;
                derDataOffset = d*m*n;
                gradientValueOffset = d;
                froNormBackward_cpu(derData + derDataOffset,data + dataOffset,gradientValue + gradientValueOffset, n*n);
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
            ptrdiff_t OutputOffset,aux_TOffset;
            T alpha;
            memcpy(output,data,sizeof(T)*length*L);
            for(d = 0; d < L; d++){
                OutputOffset = d*length;
                aux_TOffset  = d;
                alpha = sqrt(aux_T[aux_TOffset]);
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::scal(context,
                                                                     length,
                                                                     alpha,
                                                                     output + OutputOffset,ptrdiff_t(1));
                if(error != vl::VLE_Success) {goto done ;}
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
            memcpy(derData,derOutput,sizeof(T)*length*L);
            for(d = 0; d < L; d++){
                derOutputOffset = d*length;
                derData_auxOffset = d;
                dataOffset = d*length;
                aux_TOffset= d;
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::dot(context,
                                                                    length,
                                                                    data + dataOffset,ptrdiff_t(1),
                                                                    derOutput + derOutputOffset,ptrdiff_t(1),
                                                                    derData_aux + derData_auxOffset);
                if(error != vl::VLE_Success) {goto done ;}
            }
            for(d = 0;d < L; d++){
                derData_auxOffset = d;
                aux_TOffset= d;
                derData_aux[derData_auxOffset] /= 2 * pow(aux_T[aux_TOffset], 3.0f /2.0f);
            }
            for(d = 0;d < L; d++){
                derDataOffset   = d*length;
                aux_TOffset= d;
                error = vl::impl::blas<vl::VLDT_CPU, dataType>::scal(context,
                                                                     length,
                                                                     (T)(sqrt(aux_T[aux_TOffset])),
                                                                     derData + derDataOffset,ptrdiff_t(1));
                if(error != vl::VLE_Success) {goto done ;}
            }
            done:
            return context.passError(error, __func__);
        }
    };
} }

template struct vl::impl::cov_froNorm<vl::VLDT_CPU, float ,vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::cov_froNorm<vl::VLDT_CPU, double, vl::VLDT_Double> ;
#endif
