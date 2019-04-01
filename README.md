# CSE240D Final Project
# Bitfusion architecture with support for 6 bit quantization

## Introduction

This repository contains three main components - 
  * Implementation of Le-Net 5 in Keras
  * Compiler
  * Hardware Simulator

## Implementation of Le-Net 5 in Keras

This is contained in ```compiler/keras_implementation.py```. Le-Net 5 has 8 layers and we simulate the first convolution layer. ```sdfsdfdsf.py``` extracts the parameters of that convolution layer and passes it to the compiler.


## Compiler

The compiler generates instructions for the configuration parsed from the above implementation. The compiler can also be run manually through command line - 

`python gen_opcode.py --input_image_shape 1 5 5 --kernel_shape 1 2 2 --padding 0 --bitfusion_dim 4 4 --ibuf_size 256 --wbuf_size 128 --obuf_size 16384 --input_quant 6 --weight_quant 6 > /tmp/tmp`

Following is the help section if required - 

`python gen_opcode.py --help`

```
usage: gen_opcode.py [-h]
                     [--input_image_shape INPUT_IMAGE_SHAPE [INPUT_IMAGE_SHAPE ...]]
                     [--kernel_shape KERNEL_SHAPE [KERNEL_SHAPE ...]]
                     [--padding [PADDING]]
                     [--bitfusion_dim BITFUSION_DIM [BITFUSION_DIM ...]]
                     [--ibuf_size [IBUF_SIZE]] [--wbuf_size [WBUF_SIZE]]
                     [--obuf_size [OBUF_SIZE]] [--input_quant [INPUT_QUANT]]
                     [--weight_quant [WEIGHT_QUANT]]

Optional app description

optional arguments:
  -h, --help            show this help message and exit
  --input_image_shape INPUT_IMAGE_SHAPE [INPUT_IMAGE_SHAPE ...]
                        input image shape, default:10 28 28
  --kernel_shape KERNEL_SHAPE [KERNEL_SHAPE ...]
                        kernel shape, default:3 3 3
  --padding [PADDING]   padding for each input, default:0
  --bitfusion_dim BITFUSION_DIM [BITFUSION_DIM ...]
                        layout of fusion units, defaut:16 16
  --ibuf_size [IBUF_SIZE]
                        size (B) of input buffers for each fusion unit,
                        default:128
  --wbuf_size [WBUF_SIZE]
                        size (B) of weight buffers for each fusion unit,
                        default:128
  --obuf_size [OBUF_SIZE]
                        size (B) of output buffers of each column of fusion
                        units, default:1024
  --input_quant [INPUT_QUANT]
                        inputs quantization 2/4/6/8, default:8
  --weight_quant [WEIGHT_QUANT]
                        weights quantization 2/4/6/8, default:8
```

## Hardware Simulator

This is the simulator which will run the code stream emitted by the compiler in the above step. To run it, follow the steps below - 

`python reconstruct_matrix.py`

## Example flow - 

1. `python gen_opcode.py --input_image_shape 1 5 5 --kernel_shape 1 2 2 --padding 0 --bitfusion_dim 4 4 --ibuf_size 256 --wbuf_size 128 --obuf_size 16384 --input_quant 6 --weight_quant 6 | tail -2`

   **Output**
```
cycles used:4
total bitBricks used across all cycles:576/1024 = 56.25%
total fusionUnits used across all cycles:48/64 = 75.0%
```

2. `python reconstruct_matrix.py | tail -5`
    The output instructions are generated into dir `cycle_instr_dir/` and this script picks it up automatically.
   
   **Input Image**
```
input_image_shape:(1, 5, 5)
[[[255 255 255 255 255]
  [255 255 255 255 255]
  [255 255 255 255 255]
  [255 255 255 255 255]
  [255 255 255 255 255]]]
```
  **Kernel**
```
[[[63 63]
  [63 63]]]
```
  **Output convolution**
```
[[7812 7812 7812 7812]
 [7812 7812 7812 7812]
 [7812 7812 7812 7812]
 [7812 7812 7812 7812]]
```

  **Back of the envelope calculations**
  Input image and Kernel are 8b as shown above and we are doing 6b x 6b quantization, therefore, each input pixel becomes 63 and kernel pixel becomes 31. So, `63 * 31 * 4 = 7812`
