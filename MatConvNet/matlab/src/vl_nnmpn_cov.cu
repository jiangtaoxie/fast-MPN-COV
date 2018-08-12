// @file vl_nnmpn_cov.cu
// @brief MPN-COV block MEX wrapper
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2017 Peihua Li and Jiangtao Xie

All rights reserved.
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnmpn_cov.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <memory>
#include <assert.h>
#include <math.h>

enum {
  opt_epsilon = 0,
  opt_method,
  opt_regu_method,
  opt_alpha,
  opt_verbose,
  opt_cudnn,
  opt_no_cudnn,
} ;

VLMXOption  options [] = {
 {"Epsilon",       1,    opt_epsilon    },
 {"Method",        1,    opt_method     },
 {"Alpha",         1,    opt_alpha      },
 {"Regu_method",   1,    opt_regu_method},
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
  IN_DATA = 0, IN_AUX_S, IN_AUX_V, IN_AUX_D,  IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_AUX_S, OUT_AUX_V ,OUT_AUX_D, OUT_END
} ;

void mexFunction(int nout,mxArray *out[],
         int nin, mxArray const *in[])
{
   char method[128];
   char regu_method[128];
   double epsilon = 0;
   double alpha = 0.5;
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
         case opt_method :
              /* if (!vlmxIsString(optarg,-1)) {
             vlmxError(VLMXE_IllegalArgument, "METHOD is not a string.") ;
           }
           if (vlmxIsEqualToStringI(optarg, "covariance_small")) {
             //method = "covariance_small" ;
           } else if (vlmxIsEqualToStringI(optarg, "covariance")) {
             //method = "covariance" ;
           } else {
             vlmxError(VLMXE_IllegalArgument, "METHOD is not a supported method.") ;
           }*/
           break;
         case opt_epsilon :
            if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          vlmxError(VLMXE_IllegalArgument, "STRIDE is not a plain matrix.") ;
            }
            epsilon = mxGetPr(optarg)[0] ;
             break;
      case opt_alpha :
            if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
               vlmxError(VLMXE_IllegalArgument, "STRIDE is not a plain matrix.") ;
            }
             alpha = mxGetPr(optarg)[0] ;
             if(alpha <= 0){
               vlmxError(VLMXE_IllegalArgument, "ALPHA should be positive.") ;
             }
             break;
     case opt_regu_method :
              if (!vlmxIsString(optarg,-1)) {
             vlmxError(VLMXE_IllegalArgument, "METHOD is not a string.") ;
           }
           if (vlmxIsEqualToStringI(optarg, "power")) {
             //regu_method = "power" ;
           } else if (vlmxIsEqualToStringI(optarg, "mat-l2")) {
             //regu_method = "mat-l2" ;
           } else {
             vlmxError(VLMXE_IllegalArgument, "METHOD is not a supported method.") ;
           }
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
    vl::MexTensor aux_S(context);
    vl::MexTensor aux_V(context);
    vl::MexTensor aux_D(context);

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
      aux_S.init(in[IN_AUX_S]);
      aux_S.reshape(4);
      aux_V.init(in[IN_AUX_V]);
      aux_V.reshape(4);
      aux_D.init(in[IN_AUX_D]);
      aux_D.reshape(4);
    }

    if (backMode && ! vl::areCompatible(data, derOutput)) {
      mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
    }
     
     vl::TensorShape outputShape(1,1,d*(d+1)/2,n);
     vl::TensorShape aux_S_Shape(1,d,3*n,1);
     vl::TensorShape aux_V_Shape(d,d,n,1);
     vl::TensorShape aux_D_Shape(n,1,1,1);

     vl::DeviceType deviceType = data.getDeviceType() ;
     vl::DataType dataType = data.getDataType() ;
     vl::MexTensor output(context) ;
     vl::MexTensor derData(context) ;
     vl::TensorShape derDataShape(h, w, d, n);
     if (!backMode) {
          output.initWithZeros(deviceType, dataType, outputShape) ;
          aux_S.initWithZeros(deviceType, dataType, aux_S_Shape) ;
          aux_V.initWithZeros(deviceType, dataType, aux_V_Shape) ;
          aux_D.initWithZeros(vl::VLDT_CPU,dataType, aux_D_Shape);
      } else {
          derData.initWithZeros(deviceType, dataType, derDataShape) ;
     }
     
     vl::ErrorCode error ;
     if (!backMode) {
        error = vl::nnmpn_cov_forward(context,
                                      output, 
                                      data,
                                      aux_S,
                                      aux_V,
                                      aux_D,
                                      epsilon,alpha) ;
     } else {
        error = vl::nnmpn_cov_backward(context,
                                       derData, 
                                       data, 
                                       derOutput,
                                       aux_S,
                                       aux_V,
                                       aux_D,
                                       epsilon,alpha) ;
     }
     if (error != vl::VLE_Success) {
       mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
     }
    if (backMode) {
       out[OUT_RESULT] = derData.relinquish() ;
     } else {
       out[OUT_RESULT] = output.relinquish() ;
       out[OUT_AUX_S] = aux_S.relinquish() ;
       out[OUT_AUX_V] = aux_V.relinquish() ; 
       out[OUT_AUX_D] = aux_D.relinquish();
     }
}
    
