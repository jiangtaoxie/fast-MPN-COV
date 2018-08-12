function [net, info] = fast_MPN_COV_main(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%  This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%  VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.

% modified by Peihua Li  for fast MPN-COV

run(fullfile(fileparts(mfilename('fullpath')),   '..', '..', 'matlab', 'vl_setupnn.m')) ;

dataset_name = 'ILSVRC2012';  % 'ILSVRC2012' 
opts.dataDir = fullfile(vl_rootnn, 'data', dataset_name) ;
opts.modelType =  'alexnet';          %  'alexnet' 'vgg-m' 'vgg-vd-16'   'resnet-50'  'resnet-101'
           

opts.dim = 4096;  % dimension of FC layers (only for opts.modelType = 'alexnet','vgg-m' or 'vgg-vd-16')                                
opts.regu_method = 'power';       
opts.epsilon = 0;                         
opts.alpha = 0.5; 
opts.batchNormalization = true ;    

opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

if  any(strncmp(opts.modelType, {'resnet-'}, 7)) opts.networkType = 'dagnn'; end

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bn'] ; end
sfx = ['MPN-COV-' sfx '-' opts.networkType] ;
opts.expDir = fullfile(vl_rootnn, 'data', [dataset_name '-' sfx]) ;

[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12;  
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end;

fprintf('Current experimental dir is: %s\n', opts.expDir);

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
  imdb.imageDir = fullfile(opts.dataDir, 'images');
else
    switch dataset_name
        case 'ILSVRC2012'
            imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
        otherwise
             error('Unsupported database ''%s''', dataset_name) ;
    end
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end


% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  train = find(imdb.images.set == 1) ;
  if numel(train) < 5e5
      images = fullfile(imdb.imageDir, imdb.images.name(train(1:5:end))) ;
  else
      images = fullfile(imdb.imageDir, imdb.images.name(train(1:100:end))) ;
  end
  [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
                                                    'imageSize', [256 256], ...
                                                    'numThreads', opts.numFetchThreads, ...
                                                    'gpus', opts.train.gpus) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end
[v,d] = eig(rgbCovariance) ;
rgbDeviation = v*sqrt(d) ;
clear v d ;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

if isempty(opts.network)
    if any(strncmp(opts.modelType,  {'resnet-'}, 7))
          net = MPN_COV_init_resnet(opts, 'averageImage', rgbMean, 'colorDeviation', rgbDeviation, ...
                                         'classNames', imdb.classes.name, 'classDescriptions', imdb.classes.description) ;
          opts.networkType = 'dagnn' ;
    else 
        if  strcmp(opts.modelType, 'alexnet') | strcmp(opts.modelType, 'vgg-m') | strcmp(opts.modelType, 'vgg-vd-16')
              net = MPN_COV_init_simplenn(opts, imdb,  'model', opts.modelType, ...
                                  'batchNormalization', opts.batchNormalization, ...
                                  'weightInitMethod', opts.weightInitMethod, ...
                                  'networkType', opts.networkType, ...
                                  'averageImage', rgbMean, ...
                                  'colorDeviation', rgbDeviation, ...
                                  'classNames', imdb.classes.name, ...
                                  'classDescriptions', imdb.classes.description) ;
        else
                error('Error--the architecture is not supported yet! \n');
        end
                              
    end
else
  net = opts.network ;
  opts.network = [] ;
end

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainFn = @cnn_train ;
  case 'dagnn', trainFn = @cnn_train_dag ;
end

[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat')

switch opts.networkType
  case 'simplenn'
    save(modelPath, '-struct', 'net') ;
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

if numel(meta.normalization.averageImage) == 3
  mu = double(meta.normalization.averageImage(:)) ;
else
  mu = imresize(single(meta.normalization.averageImage), ...
                meta.normalization.imageSize(1:2)) ;
end

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct(...
  'useGpu', useGpu, ...
  'numThreads', opts.numFetchThreads, ...
  'imageSize',  meta.normalization.imageSize(1:2), ...
  'cropSize', meta.normalization.cropSize, ...
  'subtractAverage', mu) ;

% Copy the parameters for data augmentation
bopts.train = bopts.test ;
for f = fieldnames(meta.augmentation)'
  f = char(f) ;
  bopts.train.(f) = meta.augmentation.(f) ;
end

fn = @(x,y) getBatch(bopts,useGpu,lower(opts.networkType),x,y) ;

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;  
else
  phase = 'test' ;
end
data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
  labels = imdb.images.label(batch) ;
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end

