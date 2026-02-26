# BoostTransformer
PyTorch code implementation for "BoostTransformer: A Lightweight Transformer for Remote Sensing Image Captioning".

## Installation and Dependencies
Create the `m2` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate m2
```
## Data preparation
Download Sydney_Captions, UCM_Captions and RSICD dataset from:  
https://github.com/201528014227051/RSICD_optimal  

For `./evaluation` folder, please refer to:  
https://github.com/One-paper-luck/MG-Transformer/tree/main/evaluation

## Train
```
python train_two-stage.py
```

## Evaluate
```
python test.py
```


## Citation:
```
@ARTICLE{11392789,
  author={Bo, Tiancheng and Ma, Yuandong and Song, Qing},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={BoostTransformer: A Lightweight Transformer for Remote Sensing Image Captioning}, 
  year={2026},
  volume={23},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2026.3663867}}
```


## Reference:
We sincerely appreciate the following repository:  
https://github.com/One-paper-luck/MG-Transformer  
