## Fine-tune pre-trained model in PyTorch

By using our code, we reproduce the results of three popular fine-grained benchmarks.(i.e., Bird, Aircrafts and Cars) We will keep updating the results of this page.

#### Our experiments are running on
- [x] PyTorch 0.4 or above
- [x] 2 x 1080Ti
- [x] Cuda 9.0 with CuDNN 7.0

#### Change log
>**2019/01/22**
>> Add Compact bilinear pooling method (`CBP.py`).
>
>> Update the results of CBP.

>**2019/11/18**
>> Update the results of fast-MPN-COV-VGG-D. (`finetune_mpncovvggd16.sh`)

## Results (top-1 accuracy rates, %)

All the reproduced results use neither bounding boxes nor part annotations, and the SVM classifier is not performed.m

#### MPN-COV (ours)
<table>
     <tr>
         <th rowspan="2" style="text-align:center;">Backbone model</th>
         <th rowspan="2" style="text-align:center;">Dim.</th>
         <th colspan="2" style="text-align:center;"><a href="http://www.vision.caltech.edu/visipedia/CUB-200-2011.html">Birds</a></th>
         <th colspan="2" style="text-align:center;"><a href="http://ai.stanford.edu/~jkrause/cars/car_dataset.html">Aircrafts</a></th>
         <th colspan="2" style="text-align:center;"><a href="http://www.robots.ox.ac.uk/~vgg/data/oid/">Cars</a></th>
     </tr>
     <tr>
         <td> paper </td>
         <td> reproduce</td>
         <td> paper </td>
         <td> reproduce</td>
         <td> paper </td>
         <td> reproduce</td>
     </tr>
     <tr>
         <td style="text-align:center;">VGG-D</td>
         <td style="text-align:center;">32K</td>
         <td style="text-align:center;">87.2</td>
         <td style="text-align:center;"><b>87.0</b></td>
         <td style="text-align:center;">90.0</td>
         <td style="text-align:center;"><b>91.7</b></td>
         <td style="text-align:center;">92.5</td>
         <td style="text-align:center;"><b>93.2</b></td>
     </tr>
     <tr>
         <td style="text-align:center;">ResNet-50</td>
         <td style="text-align:center;">32K</td>
         <td style="text-align:center;">88.1</td>
         <td style="text-align:center;"><b>88.0</b></td>
         <td style="text-align:center;">90.0</td>
         <td style="text-align:center;"><b>90.3</b></td>
         <td style="text-align:center;">92.8</td>
         <td style="text-align:center;"><b>92.3</b></td>
     </tr>
     <tr>
         <td style="text-align:center;">ResNet-101</td>
         <td style="text-align:center;">32K</td>
         <td style="text-align:center;">88.7</td>
         <td style="text-align:center;">TODO</td>
         <td style="text-align:center;">91.4</td>
         <td style="text-align:center;">TODO</td>
         <td style="text-align:center;">93.3</td>
         <td style="text-align:center;">TODO</td>
     </tr>
</table>

#### Bilinear CNN
<table>
      <tr>
          <th rowspan="2" style="text-align:center;">Backbone model</th>
          <th rowspan="2" style="text-align:center;">Dim.</th>
          <th colspan="2" style="text-align:center;"><a href="http://www.vision.caltech.edu/visipedia/CUB-200-2011.html">Birds</a></th>
          <th colspan="2" style="text-align:center;"><a href="http://ai.stanford.edu/~jkrause/cars/car_dataset.html">Aircrafts</a></th>
          <th colspan="2" style="text-align:center;"><a href="http://www.robots.ox.ac.uk/~vgg/data/oid/">Cars</a></th>
      </tr>
      <tr>
          <td> paper </td>
          <td> reproduce</td>
          <td> paper </td>
          <td> reproduce</td>
          <td> paper </td>
          <td> reproduce</td>
      </tr>
     <tr>
         <td rowspan="2" style="text-align:center;">VGG-D</td>
         <td style="text-align:center;">262K</td>
         <td style="text-align:center;">84.0</td>
         <td style="text-align:center;"><b>84.0</b></td>
         <td style="text-align:center;">86.9</td>
         <td style="text-align:center;"><b>86.9</b></td>
         <td style="text-align:center;">90.6</td>
         <td style="text-align:center;"><b>90.5</b></td>
     </tr>
     </tr>
</table>

#### Compact bilinear pooling
<table>
      <tr>
          <th rowspan="2" style="text-align:center;">Backbone model</th>
          <th rowspan="2" style="text-align:center;">Dim.</th>
          <th colspan="2" style="text-align:center;"><a href="http://www.vision.caltech.edu/visipedia/CUB-200-2011.html">Birds</a></th>
          <th colspan="2" style="text-align:center;"><a href="http://ai.stanford.edu/~jkrause/cars/car_dataset.html">Aircrafts</a></th>
          <th colspan="2" style="text-align:center;"><a href="http://www.robots.ox.ac.uk/~vgg/data/oid/">Cars</a></th>
      </tr>
      <tr>
          <td> paper </td>
          <td> reproduce</td>
          <td> paper </td>
          <td> reproduce</td>
          <td> paper </td>
          <td> reproduce</td>
      </tr>
     <tr>
         <td style="text-align:center;">VGG-D</td>
         <td style="text-align:center;">8K</td>
         <td style="text-align:center;">84.0</td>
         <td style="text-align:center;"><b>83.8</b></td>
         <td style="text-align:center;">-</td>
         <td style="text-align:center;">TODO</td>
         <td style="text-align:center;">-</td>
         <td style="text-align:center;">TODO</td>
     </tr>
     </tr>
</table>

#### Global average pooling
<table>
     <tr>
         <th style="text-align:center;">Backbone model</th>
         <th style="text-align:center;">Dimension</th>
         <th style="text-align:center;"><a href="http://www.vision.caltech.edu/visipedia/CUB-200-2011.html">Birds</a></th>
         <th style="text-align:center;"><a href="http://ai.stanford.edu/~jkrause/cars/car_dataset.html">Aircrafts</a></th>
         <th style="text-align:center;"><a href="http://www.robots.ox.ac.uk/~vgg/data/oid/">Cars</a></th>
     </tr>
     <tr>
         <td style="text-align:center;">ResNet-152</td>
         <td style="text-align:center;">2K</td>
         <td rowspan="7" colspan="3" style="text-align:center;">TODO </td>
     </tr>
     <tr>
         <td style="text-align:center;">ResNet-101</td>
         <td style="text-align:center;">2K</td>
     </tr>
     <tr>
         <td style="text-align:center;">ResNet-50</td>
         <td style="text-align:center;">2K</td>
     </tr>
     <tr>
         <td style="text-align:center;">DenseNet</td>
         <td style="text-align:center;">1K</td>
     </tr>
     <tr>
         <td style="text-align:center;">Inception-v3</td>
         <td style="text-align:center;">2K</td>
     </tr>
     <tr>
         <td style="text-align:center;">VGG-D</td>
         <td style="text-align:center;">4K</td>
     </tr>
     <tr>
         <td style="text-align:center;">AlexNet</td>
         <td style="text-align:center;">4K</td>
     </tr>
</table>
