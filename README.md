# PointNet.pytorch
This repo is based on the implementation for PointNet(https://arxiv.org/abs/1612.00593) in pytorch made by fxia22. 
The model was modified into the Encoder-Decoder for the denoising, it can be found in pointnet/encoder_decoder_model.py.

It is tested with pytorch-1.0.
The tb support was added.
It currently inly supports the shapenet data + we added several other models for tests.

# Download data and running

```
git clone https://github.com/mdiephuis/pointnet.pytorch
cd pointnet.pytorch
pip install -e .
```

Download and build visualization tool
```
cd script
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
```

Training 
```
cd utils

```

# Performance
To be added

## 
# Links

- [Project Page](http://stanford.edu/~rqi/pointnet/)
- [Tensorflow implementation](https://github.com/charlesq34/pointnet)
- [The initial Pytorch version of Pointnet used as a base for this project](https://github.com/fxia22/pointnet.pytorch)
