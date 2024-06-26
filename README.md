# FishSegSSL: A Semi-supervised Semantic Segmentation Framework for Fish-eye Images

Official implementation of the paper [FishSegSSL: A Semi-supervised Semantic Segmentation Framework for Fish-eye Images](https://www.mdpi.com/2313-433X/10/3/71).

**Semi-supervised Semantic Segmentation**
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fishsegssl-a-semi-supervised-semantic/semi-supervised-semantic-segmentation-on-45)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-45?p=fishsegssl-a-semi-supervised-semantic)


## Requirements

`pip3 install -r requirements.txt`

## Training

Training can be fired with any one of the following commands:

`python3 main.py --config data/configs/params_semantic_cutmix_fixmatch_0_95_best_cutmix_cfg.yaml`


## Acknowledgement
We thank the following repos on which our codebase is built upon
- [https://github.com/valeoai/WoodScape](https://github.com/valeoai/WoodScape)
- [https://github.com/charlesCXK/TorchSemiSeg](https://github.com/charlesCXK/TorchSemiSeg)
