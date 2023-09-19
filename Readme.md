# Quantize Sequential Recommenders without Private Data

[[Paper]](https://weizhangltt.github.io/paper/WWW23_Quantize.pdf) is accepted by WWW'2023.

Author: Lingfeng Shi, Yuang Liu, Jun Wang, Wei Zhang

East China Normal University (ECNU)

## Intro

![Frame Image](/image/PRecQ.jpg)

## Requirements
*   Python 3.8
*   Pytorch>=1.8.0

## Usage
## Other Datasets and models
### keypoints
1. Training a good teacher model is essential, it determines the upperbound of the student model. 
1. Max_length
*  the scope of gumbel softmax, and 
* The rhythm of G and S training is important, and G should be trained much less frequently than S



## Reference
```
@inproceedings{shi2023quantize,
  title={Quantize Sequential Recommenders Without Private Data},
  author={Shi, Lingfeng and Liu, Yuang and Wang, Jun and Zhang, Wei},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={1043--1052},
  year={2023}
}
```
