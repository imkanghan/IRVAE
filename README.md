# Inference-Reconstruction Variational Autoencoder for Light Field Image Reconstruction
### [Paper](https://ieeexplore.ieee.org/abstract/document/9864283)

This repository is an implementation of light field image reconstruction described in the paper "Inference-Reconstruction Variational Autoencoder for Light Field Image Reconstruction". 

[Kang Han](https://imkanghan.github.io/)<sup>1</sup>, [Wei Xiang](https://scholars.latrobe.edu.au/wxiang)<sup>2</sup>

<sup>1</sup>James Cook University, <sup>2</sup>La Trobe University

### Abstract
Light field cameras can capture the radiance and direction of light rays by a single exposure, providing a new perspective to photography and 3D geometry perception. However, existing sub-aperture based light field cameras are limited by their sensor resolution to obtain high spatial and angular resolution images simultaneously. In this paper, we propose an inference-reconstruction variational autoencoder (IR-VAE) to reconstruct a dense light field image out of four corner reference views in a light field image. The proposed IR-VAE is comprised of one inference network and one reconstruction network, where the inference network infers novel views from existing reference views and viewpoint conditions, and the reconstruction network reconstructs novel views from a latent variable that contains the information of reference views, novel views, and viewpoints. The conditional latent variable in the inference network is regularized by the latent variable in the reconstruction network to facilitate information flow between the conditional latent variable and novel views. We also propose a statistic distance measurement dubbed the mean local maximum mean discrepancy (MLMMD) to enable the measurement of the statistic distance between two distributions with high-dimensional latent variables, which can capture richer information than their low-dimensional counterparts. Finally, we propose a viewpoint-dependent indirect view synthesis method to synthesize novel views more efficiently by leveraging adaptive convolution. Experimental results show that our proposed methods outperform state-of-the-art methods on different light field datasets.


### Environment
The code was tested with

- Pytorch-1.10
- CUDA-10.0.130

The adaptive convolution module will be complied at the running time. So the first time running of the code may take some time.

### Dataset

Downloading 100 scenes training dataset and 30 scenes testing dataset from [here](https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/). Following the instruction in [DistgASR](https://github.com/YingqianWang/DistgASR) to prepare the data for reconstructing the Y channel in the YCbCr color space.

### Testing
Changing the testing_data_path to your data path in the config file. The following command will produce the results (PSNR, 40.48 dB) on the 30 scenes dataset in Table III in the paper.
```
python test.py --config configs/lytro-2x2-8x8-rgb-hrnet-encoder.txt
```
Using the RDN as the encoder will produce a slightly better PSNR (40.78 dB) on the 30 scenes dataset.
```
python test.py --config configs/lytro-2x2-8x8-rgb-rdn-encoder.txt
```
The following command will produce the results (PSNR, 43.70 dB) on the 30 scenes dataset for the task of 2x2-7x7 in Table IV in the paper.
```
python test.py --config configs/lytro-2x2-7x7-y-rdn-encoder.txt
```


### Training
```
python train.py --config configs/lytro-2x2-8x8-rgb-rdn-encoder.txt
```

### Citation
If you find this code useful in your research, please cite:

    @ARTICLE{9864283,
        author={Han, Kang and Xiang, Wei},
        journal={IEEE Transactions on Image Processing},
        title={Inference-Reconstruction Variational Autoencoder for Light Field Image Reconstruction},
        year={2022},
        volume={31},
        pages={5629-5644},
        doi={10.1109/TIP.2022.3197976}
    }
