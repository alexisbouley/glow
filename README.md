# Glow

This repository implements the [Glow](https://arxiv.org/abs/1807.03039) model using PyTorch on the CelebA dataset. 
The goal was to create a denoiser from glow model.


## Setup and run

The code has minimal dependencies. You need python 3.6+ and up to date versions of:

```
pytorch (tested on 1.1.0)
torchvision
pytorch-ignite
```

You also need to [download](https://www.kaggle.com/datasets/alexisbouley/celeba) the data and put it in the CelebA folder.

**To train your own model:**

```
python main.py
```

Glow is a memory hungry model and it might be necessary to tune down the model size for your specific GPU. The output files will be send to `output/`.


## Evaluate

The [notebook](work.ipynb) computes samples from the model and also try to recover some images.


