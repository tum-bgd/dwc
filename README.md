# Dead wood image tile classification on KV260

some intro

- four classes: debris, forest, water, other

## Experimental Environment Setup

also some intro

### Host Machine

only for CUDA-enabled host machine

```
docker pull tumbgd/vai-pt-cuda
```

### Target Machine

Xilinx Kria KV260

## Usage

### On host machine

0. h5 dataset

    download link?

1. Train model

2. Pruning the trained model

3. Quantizing the pruned trained model

4. Compiling the quantized pruned trained model (for on-board depolyment):

    ```bash
    vai_c_xir -x ./quantize_result/Model_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o dwc_ob -n dwc_ob
    ```

### On target machine

Please follow the [README](./onboard/README.md) in `onboard`.

## Performances

| Device & Model          | Inference Speed (FPS) |
|:-----------------------:|:---------------------:|
| opt on KV260            | 211.07                |
| non-opt on KV260        | 87.71                 |
| non-opt on laptop (CPU) | 255.59                |
| non-opt on laptop (GPU) | 308.75                |

## Remaining Questions

1. setup VART on Ubuntu 22.04 (currently on pre-built images with shabby GUI)
    
    - lower version VART seems okay, but much less functions supported.

2. `bias_corr` is `None`. Seems no error here (accuray hardly drops), but why not 0 rather than `None`

3. `ReLU` should be supported as stated in the Xilinx document but not in practice.
    
    - Maybe `torch.nn.ReLU` is not supported, but `toerch.nn.functional.relu` is. Need a try here.
    - Or just not supported. In this case, try all possible activation functions to avoid multiple subgraphs. Our aim is to run a model fully on a single DPU graph to aviod data copy between DPU and CPU.