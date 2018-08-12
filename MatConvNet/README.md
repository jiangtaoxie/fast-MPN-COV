## Implementation details

We developed our programs based on MatConvNet and Matlab 2017b, running under either Ubuntu 14.04.5 LTS. To implement Fast MPN-COV meta-layer, we designed a loop-embedded directed graph, which can be divided into 3 sublayers, including `Post-normalization`, `Newton-Schulz iteration` and `Post-compensation`. Both the forward and backward propagations are performed using C++ on GPU.

### Created and Modified

1. Files we created to implement fast MPN-COV meta-layer

```
└── matconvnet_root_dir
    └── matlab
        ├── +dagnn
        │   ├── OBJ_ConvNet_Cov_FroNorm.m
        │   ├── OBJ_ConvNet_COV_Pool.m
        │   ├── OBJ_ConvNet_COV_ScaleFro.m
        │   ├── OBJ_ConvNet_COV_ScaleTr.m
        │   ├── OBJ_ConvNet_Cov_Sqrtm.m
        │   └── OBJ_ConvNet_Cov_TraceNorm.m
        └── src
            ├── bits
            │   ├── impl
            │   │   ├── blashelper_cpu.hpp
            │   │   ├── blashelper_gpu.hpp
            │   │   ├── cov_froNorm_cpu.cpp
            │   │   ├── cov_froNorm_gpu.cu
            │   │   ├── cov_pool_cpu.cpp
            │   │   ├── cov_pool_gpu.cu
            │   │   ├── cov_sqrtm_cpu.cpp
            │   │   ├── cov_sqrtm_gpu.cu
            │   │   ├── cov_traceNorm_cpu.cpp
            │   │   ├── cov_traceNorm_gpu.cu
            │   │   ├── nncov_froNorm_blas.hpp
            │   │   ├── nncov_pool_blas.hpp
            │   │   ├── nncov_sqrtm_blas.hpp
            │   │   └── nncov_traceNorm_blas.hpp
            │   ├── nncov_froNorm.cpp
            │   ├── nncov_froNorm.cu
            │   ├── nncov_froNorm.hpp
            │   ├── nncov_pool.cpp
            │   ├── nncov_pool.cu
            │   ├── nncov_pool.hpp
            │   ├── nncov_sqrtm.cpp
            │   ├── nncov_sqrtm.cu
            │   ├── nncov_sqrtm.hpp
            │   ├── nncov_traceNorm.cpp
            │   ├── nncov_traceNorm.cu
            │   └── nncov_traceNorm.hpp
            ├── vl_nncov_froNorm.cpp
            ├── vl_nncov_froNorm.cu
            ├── vl_nncov_pool.cpp
            ├── vl_nncov_pool.cu
            ├── vl_nncov_sqrtm.cpp
            ├── vl_nncov_sqrtm.cu
            ├── vl_nncov_traceNorm.cpp
            └── vl_nncov_traceNorm.cu

```
2. Files we modified to support Fast MPN-COV meta-layer

```
└── matconvnet_root_dir
    └── matlab
        ├── vl_compilenn.m
        └── simplenn
            └── vl_simplenn.m
```

## Installation
1. We package our programs and [demos](./examples/imagenet) in MatConvNet toolkit,you can download this [PACKAGE](https://github.com/jiangtaoxie/fast-MPN-COV/archive/master.zip) directly, or in your Terminal type:

```Ubuntu
   >> git clone https://github.com/jiangtaoxie/fast-MPN-COV
```

2. Then you can follow the tutorial of MatConvNet's [installation guide](http://www.vlfeat.org/matconvnet/install/) to complile, for example:

```matlab
   >> vl_compilenn('enableGpu', true, ...
                   'cudaRoot', '/Developer/NVIDIA/CUDA-8.0', ...
                   'cudaMethod', 'nvcc', ...
                   'enableCudnn', true, ...
                   'cudnnRoot', 'local/cudnn-rc4') ;
```
3. Currently, we use MatConvNet 1.0-beta22. For newer versions, please consult the MatConvNet [website](http://www.vlfeat.org/matconvnet).

## Usage
### Insert MPN-COV layer into your network

1. Under SimpleNN Framework

   (1). Using traceNorm
```matlab
net.layers{end+1} = struct('type','cov_pool','name','iter_cov_pool');
net.layers{end+1} = struct('type','cov_traceNorm','name','iter_cov_traceNorm');
net.layers{end+1} = struct('type','cov_sqrtm','name','iter_cov_sqrtm','coef',1,'iterNum',5);
net.layers{end+1} = struct('type','cov_traceNorm_aux','name','iter_cov_traceNorm_aux');
```
(2). Using frobeniusNorm
```matlab
net.layers{end+1} = struct('type','cov_pool','name','iter_cov_pool');
net.layers{end+1} = struct('type','cov_froNorm','name','iter_cov_froNorm');
net.layers{end+1} = struct('type','cov_sqrtm','name','iter_cov_sqrtm','coef',1,'iterNum',5);
net.layers{end+1} = struct('type','cov_froNorm_aux','name','iter_cov_froNorm_aux');
```
2. Under DagNN Framework

   (1). Using traceNorm
```matlab
name = 'cov_pool'; % Global Covariance Pooling Layer
net.addLayer(name , dagnn.OBJ_ConvNet_COV_Pool(), lastAdded.var,   name) ;
lastAdded.var = name;
name = 'cov_trace_norm'; % pre-normalization Layer by trace-Norm
name_tr =  [name '_tr'];
net.addLayer(name , dagnn.OBJ_ConvNet_Cov_TraceNorm(),   lastAdded.var,   {name, name_tr}) ;
lastAdded.var = name;
name = 'cov_Sqrtm'; % Newton-Schulz iteration Layer
net.addLayer(name , dagnn.OBJ_ConvNet_Cov_Sqrtm( 'coef', 1, 'iterNum', 5),    lastAdded.var,   {name, [name '_Y'], [name, '_Z']}) ;
lastAdded.var = name;
lastAdded.depth = lastAdded.depth * (lastAdded.depth + 1) / 2;
name = 'cov_ScaleTr'; % post-compensation Layer by trace-Norm
net.addLayer(name , dagnn.OBJ_ConvNet_COV_ScaleTr(),       {lastAdded.var, name_tr},  name) ;
lastAdded.var = name;
```
   (2). Using frobeniusNorm
```matlab
name = 'cov_pool'; % Global Covariance Pooling Layer
net.addLayer(name , dagnn.OBJ_ConvNet_COV_Pool(), lastAdded.var,   name) ;
lastAdded.var = name;
name = 'cov_fro_norm'; % pre-normalization Layer by frobenius-Norm
name_fro =  [name '_fro'];
net.addLayer(name , dagnn.OBJ_ConvNet_Cov_FroNorm(),   lastAdded.var,   {name, name_fro}) ;
lastAdded.var = name;
name = 'cov_Sqrtm'; % Newton-Schulz iteration Layer
net.addLayer(name , dagnn.OBJ_ConvNet_Cov_Sqrtm( 'coef', 1, 'iterNum', 5),    lastAdded.var,   {name, [name '_Y'], [name, '_Z']}) ;
lastAdded.var = name;
lastAdded.depth = lastAdded.depth * (lastAdded.depth + 1) / 2;
name = 'cov_ScaleFro'; % post-compensation Layer by frobenius-Norm
net.addLayer(name , dagnn.OBJ_ConvNet_COV_ScaleFro(),       {lastAdded.var, name_fro},  name) ;
lastAdded.var = name;
```

In our [demo](https://github.com/jiangtaoxie/demo/tree/master/imagenet) code, we implement MPN-COV AlexNet, VGG-M and VGG-VD under SimpleNN framework, and MPN-COV ResNet under DagNN framework.

###  Arguments descriptions

1. **`'coef'`**: It is reserved for future use. Currently, it should be set to 1.
2. **`'iterNum'`**: The number of Newton-Schulz iteration, 3 to 5 times is enough.
