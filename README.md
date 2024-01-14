
## CDKD

The pytorch implementation for "Knowledge Distillation-Based Lightweight Change Detection in High-Resolution Remote Sensing Imagery for On-Board Processing". The paper is submitted to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. 


## Requirements

- Python 3.8
- Pytorch 1.9


## Dataset format

- CDD/SYSU-CD/WHU-CD
  - train
       - A
       - B
       - OUT
  - val
       - A
       - B
       - OUT
  - test
       - A
       - B
       - OUT

## Train from scratch

    python train.py

## Evaluate model performance

    python eval.py

## Pre-trained teacher models

The pre-trained teacher models are available. 

[baidu disk](https://pan.baidu.com/s/16UmjXf_ffJZN2DZ02iAGzQ?pwd=cdkd) (cdkd)


## Citation

If you find this work valuable or use our code in your own research, please consider citing us with the following bibtex:

```
@ARTICLE{
  author={G. Wang, N. Zhang, J. Wang, W. Liu, Y. Xie and H. Chen},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Knowledge Distillation-Based Lightweight Change Detection in High-Resolution Remote Sensing Imagery for On-Board Processing}, 
  year={2024},
  volume={},
  number={},
  pages={}}
```

## Contact Information

Guoqing Wang: bit_wgq@163.com

