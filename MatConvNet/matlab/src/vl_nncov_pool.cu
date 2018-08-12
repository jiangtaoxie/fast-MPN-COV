// @file vl_nncov_pool.cu
// @brief fast MPN-COV block MEX wrapper
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nncov_pool.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <memory>
#include <assert.h>
#include <math.h>

enum {
  opt_verbose,
  opt_cudnn,
  opt_no_cudnn,
} ;

VLMXOption  options [] = {
 {"Verbose",       0,    opt_verbose    },
 {"Cudnn",         0,    opt_cudnn      },
 {"Nocudnn",       0,    opt_no_cudnn   }
} ;

vl::MexContext context ;

void atExit()
{
   context.clear();
}

enum {
  IN_DATA = 0,IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout,mxArray *out[],
         int nin, mxArray const *in[])
{
   bool backMode = false;
   int verbosity = 0;
   int opt;
   int h,w,d,n;
   int next = IN_END;
   mxArray const* optarg ;
   mexAtExit(atExit) ;
   if (nin < 1) {
     mexErrMsgTxt("The arguments are less than one.");
   }
   
   if (nin > 1 && vlmxIsString(in[1],-1)) {
      next = 1;
      backMode = 0;
    } else {
       backMode = (nin >= 2) ;
    }

   while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
      switch (opt) {
         case opt_verbose :
           ++ verbosity ;
           break;
         case opt_no_cudnn :
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(false) ;
#endif
             break ;

           case opt_cudnn :
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(false) ; // true -> false
#endif 
             break ;                        
          default:
             break ;
      }
    }
    vl::MexTensor data(context) ;
    vl::MexTensor derOutput(context) ;
    data.init(in[IN_DATA]) ;
    h = data.getHeight();
    w = data.getWidth();
    d = data.getDepth();
    n = data.getSize();
    vl::TensorShape new_dataShape(h*w, d , n ,1);
    data.reshape(new_dataShape); 
    if (backMode) {
      derOutput.init(in[IN_DEROUTPUT]) ;
      derOutput.reshape(4) ;
    }

    if (backMode && ! vl::areCompatible(data, derOutput)) {
      mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
    }
     
     vl::TensorShape outputShape(d,d,1,n);
     vl::DeviceType deviceType = data.getDeviceType() ;
     vl::DataType dataType = data.getDataType() ;
     vl::MexTensor output(context) ;
     vl::MexTensor derData(context) ;
     vl::TensorShape derDataShape(h, w, d, n);
     if (!backMode) {
          output.initWithZeros(deviceType, dataType, outputShape) ;
      } else {
          derData.initWithZeros(deviceType, dataType, derDataShape) ;
     }
     
     vl::ErrorCode error ;
     if (!backMode) {
        error = vl::nncov_pool_forward(context,
                                      output, 
                                      data) ;
     } else {
        error = vl::nncov_pool_backward(context,
                                       derData, 
                                       data, 
                                       derOutput) ;
     }
     if (error != vl::VLE_Success) {
       mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
     }
    if (backMode) {
       out[OUT_RESULT] = derData.relinquish() ;
     } else {
       out[OUT_RESULT] = output.relinquish() ;
     }
}
    
