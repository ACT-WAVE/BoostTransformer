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


## Train
```
python train_two-stage.py
```

## Evaluate
```
python test.py
```


# Citation:
```

```



## Reference:
Thanks to the following repository: https://github.com/One-paper-luck/MG-Transformer  