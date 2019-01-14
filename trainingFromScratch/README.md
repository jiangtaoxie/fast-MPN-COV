## Train from scratch

By using our code, we reproduce the results of our Fast MPN-COV ResNet models on ImageNet 2012.

#### Our experiments are running on
- [x] PyTorch 0.4 or above
- [x] 2 x 1080Ti
- [x] Cuda 9.0 with CuDNN 7.0

## Results

#### Classification results (single crop 224x224, %) on ImageNet 2012 validation set
 <table>
         <tr>
             <th rowspan="2" style="text-align:center;">Network</th>
             <th colspan="2" style="text-align:center;">Top-1 Error</th>
             <th colspan="2" style="text-align:center;">Top-5 Error</th>
         </tr>
         <tr>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reproduce</td>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reproduce</td>
         </tr>
         <tr>
             <td style="text-align:center">fast MPN-COV-ResNet50</td>
             <td style="text-align:center;">22.14</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">6.22</td>
             <td style="text-align:center;"><b>6.13</b></td>
         </tr>
         <tr>
             <td style="text-align:center">fast MPN-COV-ResNet101</td>
             <td style="text-align:center;">21.21</td>
             <td style="text-align:center;"><b>20.99</b></td>
             <td style="text-align:center;">5.68</td>
             <td style="text-align:center;"><b>5.56</b></td>
         </tr>

</table>
