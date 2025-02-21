# Test-time Adaptation for Regression by Subspace Alignment (ICLR 2025)
The official implementation for ICLR 2025 paper "Test-time Adaptation for Regression by Subspace Alignment."  
[[OpenReview](https://openreview.net/forum?id=SXtl7NRyE5)] [[arXiv](https://arxiv.org/abs/2410.03263)]

![overview](overview.png)

## 0. Environment
- Prepare the datasets (SVHN, MNIST, UTKFace, Biwi Kinect, California Housing) and write their path in `dataset/dataset_config.py`.
- Install dependencies or build the docker image according to `docker/Dockerfile`.

```bash
$ docker build -t tta_regression docker --no-cache
```


## 1. Training the source model
```bash
$ python3 train_source.py -c configs/train_source/yaml.json -o result/source/svhn

# running with the docker image
$ docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) --gpus device=0 tta_regression python3 train_source.py -c configs/train_source/svhn.yaml -o result/source/svhn
```


## 2. Computing the feature statistics
```bash
$ python3 feature_stats.py -c configs/feature_stats/svhn.yaml -o result/source/svhn
```

The pre-trained model and feature statistics for SVHN are available in `result/`.


## 3. TTA
```bash
$ python3 adaptation.py -c configs/tta/svhn.yaml -o result/tta/svhn
```


## Citation
If our work assists your research, please cite our paper:

```
@inproceedings{adachi2025testtime,
title={Test-time Adaptation for Regression by Subspace Alignment},
author={Kazuki Adachi and Shin'ya Yamaguchi and Atsutoshi Kumagai and Tomoki Hamagami},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=SXtl7NRyE5}
}
```
