classdef OBJ_ConvNet_COV_Pool < dagnn.Layer
  properties
    opts = {'cuDNN'}
    cudnn = {'CuDNN'} 
  end


  
  methods
    function outputs = forward(self, inputs, params)
               [outputs{1}] =  vl_nncov_pool(inputs{1}, self.cudnn{:});
    end
   
     function [derInputs, derParams] = backward(self, inputs, params, derOutputs, outputs)
               [derInputs{1}] = vl_nncov_pool(inputs{1}, derOutputs{1}, self.cudnn{:});
         derParams  = {} ;
    end
    
    function obj = OBJ_ConvNet_COV_Pool(varargin)
      obj.load(varargin) ;
    end
  end
end
