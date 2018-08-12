function net = MPN_COV_init_simplenn(opts, imdb, varargin)
% CNN_IMAGENET_INIT  Initialize a standard CNN for ImageNet

% modified by Peihua Li  for MPN-COV

opts.scale = 1 ;
opts.initBias = 0 ;
opts.weightDecay = 1 ;
%opts.weightInitMethod = 'xavierimproved' ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts = vl_argparse(opts, varargin) ;


% Define layers
switch opts.modelType
  case 'alexnet'
    net.meta.normalization.imageSize = [227, 227, 3] ;
    net = alexnet_cov_pooling(net, opts)
    bs = 100;
  case {'vgg-m'}
    net.meta.normalization.imageSize = [224, 224, 3] ;
    net = vgg_m_cov_pooling(net, opts) ;
    bs = 8;
  case 'vgg-vd-16'
    net.meta.normalization.imageSize = [224, 224, 3] ;
    net = vgg_vd_16_cov_pooling(net, opts) ;
    bs  = 2;
  otherwise
    error('Unknown model ''%s''', opts.modelType) ;
end

% final touches
switch lower(opts.weightInitMethod)
  case {'xavier', 'xavierimproved'}
    net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
end
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

% Meta parameters
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ; 
net.meta.normalization.averageImage = opts.averageImage ;
net.meta.classes.name = opts.classNames ;
net.meta.classes.description = opts.classDescriptions;
net.meta.augmentation.jitterLocation = true ;
net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;  
net.meta.augmentation.jitterAspect = [2/3, 3/2] ;
if  any(strcmp(opts.modelType, {'vgg-vd-16'})) | any(strcmp(opts.modelType, {'vgg-vd-16'})) 
    net.meta.augmentation.jitterScale   =  [0.5, 1];               % randomly scaling images on [256, 512] 
end


if ~opts.batchNormalization
  lr = logspace(-2, -4, 60) ;
else
  lr = logspace(-1.2, -5.2, 20) ;
end


net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = bs ;
net.meta.trainOpts.weightDecay = 0.0005 ;

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
                 {'prediction','label'}, 'top1err') ;
    net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
                                       'opts', {'topK',5}), ...
                 {'prediction','label'}, 'top5err') ;
  otherwise
    assert(false) ;
end

%% original architecture
% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad)
% --------------------------------------------------------------------

info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), ...
                             ones(out, 1, 'single')*opts.initBias}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'dilate', 1, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), ...
                               zeros(out, 2, 'single')}}, ...
                             'epsilon', 1e-4, ...
                             'learningRate', [2 1 0.1], ...
                             'weightDecay', [0 0]) ;
end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;


% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'normalize', ...
                             'name', sprintf('norm%s', id), ...
                             'param', [5 1 0.0001/5 0.75]) ;
end

% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end


%%  %added by lph 
% --------------------------------------------------------------------
function net = vgg_vd_16_cov_pooling(net, opts)  % only consider vgg_vd_16 
% --------------------------------------------------------------------

net.layers = {} ;
net = add_block(net, opts, '1_1', 3, 3, 3, 64, 1, 1) ;
net = add_block(net, opts, '1_2', 3, 3, 64, 64, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '2_1', 3, 3, 64, 128, 1, 1) ;
net = add_block(net, opts, '2_2', 3, 3, 128, 128, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '3_1', 3, 3, 128, 256, 1, 1) ;
net = add_block(net, opts, '3_2', 3, 3, 256, 256, 1, 1) ;
net = add_block(net, opts, '3_3', 3, 3, 256, 256, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool3', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '4_1', 3, 3, 256, 512, 1, 1) ;
net = add_block(net, opts, '4_2', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '4_3', 3, 3, 512, 512, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool4', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;


net = add_block(net, opts, '5_1',     3, 3, 512, 512, 1, 1) ; 
net = add_block(net, opts, '5_2',     3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5_3_0', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5_3',     1, 1, 512, 256, 1, 0) ; % one 1x1 convolutional layer is added for dimensionality reduction

feature_dim = size(net.layers{find(cellfun(@(x) strcmp(x.name, 'conv5_3'), net.layers), 1, 'last')}.weights{1}, 4);       
MPN_COV_layer_dim = feature_dim * (feature_dim + 1) / 2; 

switch opts.regu_method

   case {'power'}
            net.layers{end+1} = struct('type','mpn_cov','name','cov_pool','method',[],...
                                                          'regu_method', opts.regu_method, ...
                                                          'epsilon', opts.epsilon, ...
                                                          'alpha', opts.alpha);
     otherwise
                error('Unknown regu_method: only power is supported. ''%s''', opts.regu_method) ;
end


net = add_block(net, opts, '6', 1, 1, MPN_COV_layer_dim, opts.dim, 1, 0) ;
net = add_dropout(net, opts, '6') ;


net = add_block(net, opts, '7', 1, 1, opts.dim, opts.dim, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, opts.dim, numel(opts.classNames), 1, 0) ;

net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end




% --------------------------------------------------------------------
function net = vgg_m_cov_pooling(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;
net = add_block(net, opts, '1', 7, 7, 3, 96, 2, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '2', 5, 5, 96, 256, 2, 1) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;


net = add_block(net, opts, '3', 3, 3, 256, 512, 1, 1) ;   
net = add_block(net, opts, '4', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5-0', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5', 1, 1, 512, 256, 1, 0) ; % one 1x1 convolutional layer is added for dimensionality reduction


feature_dim = size(net.layers{find(cellfun(@(x) strcmp(x.name, 'conv5'), net.layers), 1, 'last')}.weights{1}, 4); 
MPN_COV_layer_dim = feature_dim * (feature_dim + 1) / 2; 


switch opts.regu_method

   case {'power'}
                                          
            net.layers{end+1} = struct('type','mpn_cov','name','cov_pool','method',[],...
                                                          'regu_method', opts.regu_method, ...
                                                          'epsilon', opts.epsilon, ...
                                                          'alpha', opts.alpha);
     otherwise
                error('Unknown regu_method: only power is supported. ''%s''', opts.regu_method) ;
end

net = add_block(net, opts, '6', 1, 1, MPN_COV_layer_dim, opts.dim, 1, 0) ;
net = add_dropout(net, opts, '6') ;


net = add_block(net, opts, '7', 1, 1, opts.dim, opts.dim, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, opts.dim, numel(opts.classNames), 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end


function net = alexnet_cov_pooling(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;

net = add_block(net, opts, '1', 11, 11, 3, 96, 4, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;


net = add_block(net, opts, '2', 5, 5, 48, 256, 1, 2) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;


net = add_block(net, opts, '3', 3, 3, 256, 384, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 192, 384, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 192, 256, 1, 1) ;


feature_dim = size(net.layers{find(cellfun(@(x) strcmp(x.name, 'conv5'), net.layers), 1, 'last')}.weights{1}, 4);   
MPN_COV_layer_dim = feature_dim * (feature_dim + 1) / 2; 

switch opts.regu_method
    case {'trace-norm-sqrtm'}
         net.layers{end+1} = struct('type','cov_pool','name','iter_cov_pool');
         net.layers{end+1} = struct('type','cov_traceNorm','name','iter_cov_traceNorm');
         net.layers{end+1} = struct('type','cov_sqrtm','name','iter_cov_sqrtm','coef',1,'iterNum',5);
         net.layers{end+1} = struct('type','cov_traceNorm_aux','name','iter_cov_traceNorm_aux');
    case {'fro-norm-sqrtm'}
         net.layers{end+1} = struct('type','cov_pool','name','iter_cov_pool');
         net.layers{end+1} = struct('type','cov_froNorm','name','iter_cov_froNorm');
         net.layers{end+1} = struct('type','cov_sqrtm','name','iter_cov_sqrtm','coef',1,'iterNum',5);
         net.layers{end+1} = struct('type','cov_froNorm_aux','name','iter_cov_froNorm_aux');
     otherwise
                error('Unknown regu_method: only power is supported. ''%s''', opts.regu_method) ;
end

net = add_block(net, opts, '6', 1, 1, MPN_COV_layer_dim, opts.dim, 1, 0) ;
net = add_dropout(net, opts, '6') ;


net = add_block(net, opts, '7', 1, 1, opts.dim, opts.dim, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, opts.dim, numel(opts.classNames), 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end



