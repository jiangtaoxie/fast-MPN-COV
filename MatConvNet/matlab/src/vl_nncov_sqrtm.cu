// @file vl_nncov_sqrtm.cu
// @brief fast MPN-COV block MEX wrapper
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nncov_sqrtm.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <memory>
#include <assert.h>
#include <math.h>

enum {
  opt_coef    = 0,
  opt_iterNum,
  opt_verbose,
  opt_cudnn,
  opt_no_cudnn,
} ;

VLMXOption  options [] = {
 {"Coef",          1,    opt_coef       },
 {"IterNum",       1,    opt_iterNum    },
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
  IN_DATA = 0, IN_AUX_Y, IN_AUX_Z,IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_AUX_Y, OUT_AUX_Z , OUT_END
} ;

void mexFunction(int nout,mxArray *out[],
         int nin, mxArray const *in[])
{
   int coef = 30;
   int iterNum = 5;
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
         case opt_iterNum :
            if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          vlmxError(VLMXE_IllegalArgument, "iterNum is not a plain matrix.") ;
            }
            iterNum = (int)mxGetPr(optarg)[0] ;
			if (iterNum <= 0){
				vlmxError(VLMXE_IllegalArgument, "ITERNUM should be positive.") ;
			}
             break;
		 case opt_coef :
            if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          vlmxError(VLMXE_IllegalArgument, "COEF is not a plain matrix.") ;
            }
            coef = (int)mxGetPr(optarg)[0] ;
			if (coef <= 0){
				vlmxError(VLMXE_IllegalArgument, "COEF should be positive.") ;
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
    vl::MexTensor aux_Y(context);
    vl::MexTensor aux_Z(context);
    data.init(in[IN_DATA]) ;
    h = data.getHeight();
    w = data.getWidth();
    d = data.getDepth();
    n = data.getSize();
    if (backMode) {
      derOutput.init(in[IN_DEROUTPUT]) ;
      derOutput.reshape(4) ;
      aux_Y.init(in[IN_AUX_Y]);
      aux_Y.reshape(4);
      aux_Z.init(in[IN_AUX_Z]);
      aux_Z.reshape(4);
    }

    if (backMode && ! vl::areCompatible(data, derOutput)) {
      mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
    }
     
	 vl::TensorShape outputShape(1,1,w*(w+1)/2,n);
	 vl::TensorShape aux_Y_Shape(w,w,iterNum,n);
	 vl::TensorShape aux_Z_Shape(w,w,iterNum,n);
     vl::DeviceType deviceType = data.getDeviceType() ;
     vl::DataType dataType = data.getDataType() ;
     vl::MexTensor output(context) ;
     vl::MexTensor derData(context) ;
     vl::TensorShape derDataShape(h, w, d, n);
     if (!backMode) {
          output.initWithZeros(deviceType, dataType, outputShape) ;
          aux_Y.initWithZeros(deviceType, dataType, aux_Y_Shape) ;
          aux_Z.initWithZeros(deviceType, dataType, aux_Z_Shape) ;
      } else {
          derData.initWithZeros(deviceType, dataType, derDataShape) ;
     }
     
     vl::ErrorCode error ;

     if (!backMode) {
        error = vl::nncov_sqrtm_forward(context,
                                      output, 
                                      data,
                                      aux_Y,
                                      aux_Z,
									  coef,
									  iterNum) ;
     } else {
        error = vl::nncov_sqrtm_backward(context,
                                       derData, 
                                       data, 
                                       derOutput,
                                       aux_Y,
                                       aux_Z,
									   coef,
									   iterNum) ;
     }
     if (error != vl::VLE_Success) {
       mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
     }
    if (backMode) {
       out[OUT_RESULT] = derData.relinquish() ;
     } else {
       out[OUT_RESULT] = output.relinquish() ;
       out[OUT_AUX_Y] = aux_Y.relinquish() ;
       out[OUT_AUX_Z] = aux_Z.relinquish() ; 
     }
}
    
