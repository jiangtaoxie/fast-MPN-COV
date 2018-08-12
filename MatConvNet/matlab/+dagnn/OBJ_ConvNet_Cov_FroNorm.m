                                                                                                                              classdef OBJ_ConvNet_Cov_FroNorm < dagnn.Layer
  properties
    opts = {'cuDNN'}
    cudnn = {'CuDNN'} 
 end

  methods
    function outputs = forward(self, inputs, params)
        [outputs{1}, outputs{2}] = vl_nncov_froNorm(inputs{1},self.cudnn{:}); % xjt 2017/10/18 12:10
   end
     %% backwardAdvanced is modified slightly for custom layer
    function backwardAdvanced(obj, layer)
    %BACKWARDADVANCED Advanced driver for backward computation
    %  BACKWARDADVANCED(OBJ, LAYER) is the advanced interface to compute
    %  the backward step of the layer.
    %
    %  The advanced interface can be changed in order to extend DagNN
    %  non-trivially, or to optimise certain blocks.
      in = layer.inputIndexes ;
      out = layer.outputIndexes ;
      par = layer.paramIndexes ;
      net = obj.net ;

      inputs = {net.vars(in).value} ;
      derOutputs = {net.vars(out).der} ;  
      % for i = 1:numel(derOutputs)  % lph 2017/02/5 17:12
      %   if isempty(derOutputs{i}), return ; end
      % end
      outputs = {net.vars(out).value};  
      if isempty(derOutputs{1}), return; end

      if net.conserveMemory
        % clear output variables (value and derivative)
        % unless precious
        for i = out
          if net.vars(i).precious, continue ; end
          net.vars(i).der = [] ;
          net.vars(i).value = [] ;
        end
      end

      % compute derivatives of inputs and paramerters
      % lph 2017/02/05 10:37
      % [derInputs, derParams] = obj.backward(inputs, {net.params(par).value}, derOutputs) ;
      [derInputs, derParams] = obj.backward(inputs, {net.params(par).value}, derOutputs, outputs) ;
      if ~iscell(derInputs) || numel(derInputs) ~= numel(in)
        error('Invalid derivatives returned by layer "%s".', layer.name);
      end

      % accumuate derivatives
      for i = 1:numel(in)
        v = in(i) ;
        if net.numPendingVarRefs(v) == 0 || isempty(net.vars(v).der)
          net.vars(v).der = derInputs{i} ;
        elseif ~isempty(derInputs{i})
          net.vars(v).der = net.vars(v).der + derInputs{i} ;
        end
        net.numPendingVarRefs(v) = net.numPendingVarRefs(v) + 1 ;
      end

      for i = 1:numel(par)
        p = par(i) ;
        if (net.numPendingParamRefs(p) == 0 && ~net.accumulateParamDers) ...
              || isempty(net.params(p).der)
          net.params(p).der = derParams{i} ;
        else
          net.params(p).der = vl_taccum(...
            1, net.params(p).der, ...
            1, derParams{i}) ;
          %net.params(p).der = net.params(p).der + derParams{i} ;
        end
        net.numPendingParamRefs(p) = net.numPendingParamRefs(p) + 1 ;
        if net.numPendingParamRefs(p) == net.params(p).fanout
          if ~isempty(net.parameterServer) && ~net.holdOn
            net.parameterServer.pushWithIndex(p, net.params(p).der) ;
            net.params(p).der = [] ;
          end
        end
      end
    end
    
    function [derInputs, derParams] = backward(self, inputs, params, derOutputs, outputs)
        [derInputs{1}] = vl_nncov_froNorm(inputs{1}, outputs{2}, derOutputs{1}, derOutputs{2}, self.cudnn{:});
             derParams  = {} ;
    end
    
     function obj = OBJ_ConvNet_Cov_FroNorm(varargin)
      obj.load(varargin) ;
    end
  end
end
