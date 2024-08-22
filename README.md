# EnMos_debias

This project aims to show the official PyTorch implementation for our XXX paper:


If you find this code useful, consider citing our work:
```
@inproceedings{kang2019decoupling,
  title={Decoupling representation and classifier for long-tailed recognition},
  author={Kang, Bingyi and Xie, Saining and Rohrbach, Marcus and Yan, Zhicheng
          and Gordo, Albert and Feng, Jiashi and Kalantidis, Yannis},
  booktitle={Eighth International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

## Abstract

Ensuring that model determination is independent of the training data distribution is essential for the safe deployment of machine learning models in real-world applications. Although many current methods are effective, they rely heavily on softmax-based scores. This reliance becomes problematic when the probabilities of correct and incorrect predictions are similar, leading to comparable loss values and limiting the model's ability to distinguish between prediction qualities. To address this, we propose a residual-energy-based score from an energy perspective, which enhances the differentiation of prediction quality. Additionally, we offer a principled approach to managing long-tailed problems by adapting estimation techniques from causal inference. This method yields unbiased performance estimates despite data bias. We theoretically and empirically demonstrate the robustness of this approach, showing that it is both practical and scalable.

## Installation
Most of the requirements of this projects are exactly the same as [DECOUPLING](https://github.com/facebookresearch/classifier-balancing).
### Requirements:
- Python == 3.7
- PyTorch >= 1.10.1 (Mine 1.10.1 (CUDA 11.7))
- torchvision >= 0.11.2 (Mine 0.11.2 (CUDA 11.7))
- [yaml](https://pyyaml.org/wiki/PyYAMLDocumentation)


### Dataset
- ImageNet_LT
  * Download the [ImageNet_2014](http://image-net.org/index)

- iNaturalist 2018

  * Download the dataset following [here](https://github.com/visipedia/inat_comp/tree/master/2018).
  * `cd data/iNaturalist18`, Generate image name files with this [script](data/iNaturalist18/gen_lists.py) or use the existing ones [[here](data/iNaturalist18)].

- Cifar_10/100_LT
  * 
Change the `data_root` in `main.py` accordingly.
  
### Training
The following command can be used to train models on different datasets.
- ImageNet_LT
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10001 --nproc_per_node=4 main.py --cfg ./config/ImageNet_LT/EnDebias_debias.yaml
```
where ```CUDA_VISIBLE_DEVICES``` and ```--nproc_per_node``` represent the id of GPUs and number of GPUs you use, ```--cfg``` means the config we use, where you can change other parameters, such as dataset, backbone, batach size and so on. This command shows how to training ResNext-50 on ImageNet_LT. 

### Test
The following command can be used to evaluate the trained model.
- ImageNet_LT
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10001 --nproc_per_node=4 main.py --cfg ./config/ImageNet_LT/EnDebias_debias.yaml --test
```

## Results

Example (balanced) test accuracies obtained by running this code on GPU devices is shown below for different datasets.

<center>

|              | epoches |  Many  |  Med   |  Few  | All  |                                              model                                               |  log   |
|:-------------|:-------:|:------:|:------:|:-----:|:----:|:------------------------------------------------------------------------------------------------:|:------:|
| ImageNet-LT  |   90    |  72.8  |  61.3  | 42.1  | 63.1 |[ResNeXt50](https://drive.google.com/file/d/1o4B2lHeJZ1NtGYlF6fTexaqWsjSBF-LZ/view?usp=drive_link)|[link](https://drive.google.com/file/d/1OtRqfC1zM-lFeGzHFzARY4WzCT9I77-T/view?usp=drive_link)|
| ImageNet-LT  |   180   |  74.6  |  60.3  | 42.7  | 63.4 |[ResNeXt50](https://drive.google.com/file/d/1wBJ_S6O7dLcAVWDLa_Tf2ZyWs5zPx235/view?usp=drive_link)|[link](https://drive.google.com/file/d/1KCxsNSWXEgMbVRHSNROOxu93FGfZFJxa/view?usp=drive_link)|
| ImageNet-LT  |   200   |  74.4  |  61.0  | 44.5  | 63.9 |[ResNeXt50](https://drive.google.com/file/d/1JAxJRiPXUSdOEWV-O4xuv2EsuBvbFxsc/view?usp=drive_link)|[link](https://drive.google.com/file/d/1XFw7-4JA2ZkCQU14SqYZMtIz5L9r6-Va/view?usp=drive_link)|
| iNaturalist18 |   90    |  76.7  |  75.8  | 76.0  | 76.0 |[ResNet50](https://drive.google.com/file/d/1cGAAN6qAfyF0BUHfQzP3GT3gRUEmJOOo/view?usp=drive_link) |[link](https://drive.google.com/file/d/1jkaesSWYYZHNEkJsYYj7Irg4VeZv8wsT/view?usp=drive_link)|
| iNaturalist18 |   200   |  78.8  |  78.3  | 78.2  | 78.3 |[ResNet50](https://drive.google.com/file/d/1G1hBGA9vKxEJWGYdhtZW_4oKDgo7srdR/view?usp=drive_link) |[link](https://drive.google.com/file/d/1htAX9qvNFXOPJMuB_i07aIwdnYxCH8c0/view?usp=drive_link)|
| iNaturalist18 |   400   | 80.6   | 79.6   | 79.1  | 79.5 |[ResNet50](https://drive.google.com/file/d/1pzc5quKj15hDhSfOTSB2XUuAWmRE2nAc/view?usp=drive_link) |[link](https://drive.google.com/file/d/1AGmAtCaVnWoYUBiHBSFpVKLzoWdEYFVW/view?usp=drive_link)|

</center>



<center>

| Imbalance ratio |  100  |  50   | 
|:----------------|:-----:|:-----:|
| CIFAR-10 LT     | 88.31 | 91.51 |   
| CIFAR-100 LT    | 56.3  | 62.26 |   

</center>

