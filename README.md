# SuperpixelGraph: Semi-automatic generation of building footprint through semantic-sensitive superpixel and neural graph networks

---

This repository provides a [PyTorch](https://github.com/pytorch/pytorch) implementation of semantic-sensitive superpixel generation network in our work [SuperpixelGraph](https://arxiv.org/abs/2304.05661).

## Requirements

---

The codes were developed and tested mainly with the following dependencies:

```
gdal=3.3.2
opencv=4.5.3
python=3.9
pytorch=1.9.1
pytorch-lightning=1.6.0
scikit-image=1.0.2
```

## Running the code

---

### Training

Use the following command to train model on WHU dataset:

```
python spn_train.py --data_dir /path/to/WHU/ --output_dir /path/to/outputs/ --dataset WHU --num_epochs 300
```

### Testing and evaluation

Use the following command to run model on test set of WHU:

```
python spn_test.py --model_path ./ckpt/superpixel_net_downsize16_whu.ckpt --data_dir /path/to/WHU/ --output_dir /path/to/outputs/
```

or run model on a single image:

```
python spn_demo.py --model_path /path/to/model --image_path /path/to/image --output_dir /path/to/output/
```

If you want to adjust the number of superpixels, use options `-H` and `-W` to set the height and width of images inputted to the model. Larger height and width you set, more superpixels will be generated. Note that the height or width should be a multiple of 16.

Tools provided by [superpixel-benchmark](https://github.com/davidstutz/superpixel-benchmark) are used to evaluate the testing results.
