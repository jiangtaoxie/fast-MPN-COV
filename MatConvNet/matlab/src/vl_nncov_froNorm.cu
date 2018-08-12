// @file vl_nnfroNorm.cu
// @brief fast MPN-COV block MEX wrapper
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nncov_froNorm.hpp"

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
  IN_DATA = 0, IN_AUX_T,IN_DEROUTPUT,IN_DEROUTPUT_2, IN_END
} ;

enum {
  OUT_RESULT = 0,OUT_AUX, OUT_END
} ;

void mexFunction(int nout,mxArray *out[],
         int nin, mxArray const *in[])
{
   bool backMode = false;
   bool auxMode = true;
   int verbosity = 0;
   int opt;
   int h,w,d,n;
   int next = IN_END;
   mxArray const* optarg ;
   mexAtExit(atExit) ;
   if (nin < 1) {
     mexErrMsgTxt("The arguments are less than one.");
   }
   switch (nin)
   {
   case 2:
	   backMode = false;
	   auxMode  = false;
	   next = 1;
	   break;
   case 3:
	   backMode = false;
	   auxMode  = true;
	   next = 2;
	   break;
   case 4:
	   backMode = true;
	   auxMode  = true;
	   next = 3;
	   break;
   case 5:
	   backMode = true;
	   auxMode  = false;
	   next = 4;
	   break;
   default:
	   mexErrMsgTxt("Invalid Input parameters.");
	   break;
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
	vl::MexTensor derOutput_aux(context);
	vl::MexTensor aux_T(context);

	
    data.init(in[IN_DATA]) ;
    h = data.getHeight();
    w = data.getWidth();
    d = data.getDepth();
    n = data.getSize(); 
    if (backMode) {
      derOutput.init(in[IN_DEROUTPUT]) ;
      derOutput.reshape(4) ;
	  aux_T.init(in[IN_AUX_T]);
      aux_T.reshape(4);
//	  derData.init(in[IN_DEROUTPUT]);
//	  derData.reshape(4);
	  if ( !auxMode) {
		  derOutput_aux.init(in[IN_DEROUTPUT_2]) ;
          derOutput_aux.reshape(4) ;
	  }
    } 

    if (backMode && ! vl::areCompatible(data, derOutput)) {
      mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
    }
     
     vl::TensorShape outputShape(h,w,d,n);
	 vl::TensorShape aux_T_Shape(1,1,1,n);
	 vl::TensorShape derData_auxShape(1,1,1,n);
     vl::DeviceType deviceType = data.getDeviceType() ;
     vl::DataType dataType = data.getDataType() ;
     vl::MexTensor output(context) ;
	 vl::MexTensor derData(context) ;
	 vl::MexTensor derData_aux(context);
     if (!backMode) {
//          output.init(in[IN_DATA]) ;
		 output.initWithZeros(deviceType, dataType, data.getShape());
		  if ( auxMode ){
			  aux_T.init(in[IN_AUX_T]);
              aux_T.reshape(4);
		  } else {
              aux_T.initWithZeros(deviceType, dataType, aux_T_Shape) ;
		  }
     } else {
		 derData.initWithZeros(deviceType, dataType, derOutput.getShape());
		 if (auxMode){
		  derData_aux.initWithZeros(deviceType, dataType, derData_auxShape);
	  }
	 }
     
     vl::ErrorCode error ;
     if (!backMode) {
		 if (!auxMode ){
             error = vl::nncov_froNorm_forward(context,
                                                 output, 
                                                 data,
									             aux_T) ;
		 } else {
			 error = vl::nncov_froNorm_aux_forward(context,
				                                     output,
													 data,
													 aux_T);
		 }
     } else { 
		 if ( !auxMode ) {
             error = vl::nncov_froNorm_backward(context,
                                                  derData, 
                                                  data, 
                                                  derOutput,
												  derOutput_aux,
	             							      aux_T) ;
		 } else {
			 error = vl::nncov_froNorm_aux_backward(context,
				                                      derData,
													  derData_aux,
													  data,
													  derOutput,
													  aux_T);
		 }
     }
     if (error != vl::VLE_Success) {
       mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
     }
    if (backMode) {
       out[OUT_RESULT] = derData.relinquish() ;
	   if ( auxMode) {
		   out[OUT_AUX] = derData_aux.relinquish() ;
	   }
     } else {
       out[OUT_RESULT] = output.relinquish() ; 
	   out[OUT_AUX] = aux_T.relinquish() ; 
     }
}
    
