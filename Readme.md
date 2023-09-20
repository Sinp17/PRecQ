# Quantize Sequential Recommenders without Private Data

[[Paper]](https://weizhangltt.github.io/paper/WWW23_Quantize.pdf) is accepted by WWW'2023.

Author: Lingfeng Shi, Yuang Liu, Jun Wang, Wei Zhang

East China Normal University (ECNU)

## Intro
Overview of framework PRecQ
![Frame Image](/image/PRecQ.jpg)

## Requirements
*   Python 3.8
*   Pytorch>=1.8.0

## Usage
First you need a pretrained model, code is not provided here. Then you preprocess the eval data in [gen_test_data.py](https://github.com/Sinp17/PRecQ/blob/main/data/gen_test_data.py).

After that you set paths and models in the [config.py](https://github.com/Sinp17/PRecQ/blob/main/train/config.py), and finally run: 
```
python main.py
```
## Datasets and models
Many sequential models and quantization methods are free to combine with PRecQ. 

For other datasets in our experiments could be find here: [[Foursquare-Tokyo]](https://www.kaggle.com/datasets/chetanism/foursquare-nyc-and-tokyo-checkin-dataset)  [[Steam]](https://www.kaggle.com/datasets/tamber/steam-video-games) 

## Notes
Since many hyperparameters are involved in our experiment, here we select some keypoints in choosing them:
* Training a good teacher model is essential, it determines the upperbound of the student model. 
* The scope of gumbel softmax trick, and maximum length of generated sequence are important hyperparameters, they both should not be too large.
* The rhythm of generator and student model training is important, and the generator should be trained much less frequently than the student model.



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
