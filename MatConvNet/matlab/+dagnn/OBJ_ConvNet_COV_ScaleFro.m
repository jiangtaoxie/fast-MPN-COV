classdef OBJ_ConvNet_COV_ScaleFro < dagnn.Layer
  properties
    opts = {'cuDNN'}
    cudnn = {'CuDNN'} 
  end


  
  methods
       
    function outputs = forward(self, inputs, params)
        [outputs{1}] =  vl_nncov_froNorm(inputs{1},inputs{2},self.cudnn{:}); % xjt 2017/10/18 12:09
    end
   
     function [derInputs, derParams] = backward(self, inputs, params, derOutputs, outputs)
         [derInputs{1}, derInputs{2}] = vl_nncov_froNorm(inputs{1},...   
                                                         inputs{2},...
                                                         derOutputs{1},self.cudnn{:}); % xjt 2017/10/18 12:08
         derParams  = {} ;
    end
    
    function obj = OBJ_ConvNet_COV_ScaleFro(varargin)
      obj.load(varargin) ;
    end
  end
end
