# pLM on AWS Neuron Devices Benchmark Results

# Index 📋

1. [Introduction](#introduction) 📌
2. [Benchmark Setup](#benchmark-setup) 🛠️
3. [Benchmark Tasks](#benchmark-tasks) 📊
4. [Benchmark Results](#benchmark-results) 📈
    - 4.1 [Training Results](#training-results) 🚀
        - 4.1.1 [GPU](#gpu) 💻
        - 4.1.2 [AWS Neuron Device](#aws-neuron-device) 🧠
    - 4.2 [Inference Results](#inference-results) 🧠
        - 4.2.1 [GPU](#gpu-1) 💻
        - 4.2.2 [AWS Neuron Device](#aws-neuron-device-1) 🚀
5. [Discussion](#discussion) 💬
6. [Conclusion](#conclusion) 🎉
7. [Recommendations](#recommendations) 📝
8. [Acknowledgments](#acknowledgments) 🙏
9. [References](#references) 🔍

## Introduction
This document presents the benchmark results of some of most common Protein Language Models (PLM) on two different devices: GPU and AWS Neuron Devices. The benchmark includes training and inference tasks using Protein Language Models.

The models are:

| Model                	| #Params       	| Attention Heads 	|
|----------------------	|---------------	|-----------------	|
| protBERT             	| 419.933.186   	| 16              	|
| prot_t5_xl_uniref50  	| 2.820.144.130 	| 32              	|
| esm1b_t33_650M_UR50S 	| 652.359.063   	| 33              	|
| esm2_t6_8M_UR50D     	| 7.840.763     	| 6               	|
| esm2_t12_35M_UR50D   	| 33.993.843    	| 12              	|
| esm2_t30_150M_UR50D  	| 148.796.763   	| 30              	|
| esm2_t33_650M_UR50D  	| 652.356.503   	| 33              	|
| esm2_t36_3B_UR50D    	| 2,841,632,163 	| 36              	|
| esm2_t48_15B_UR50D   	|               	|                 	|

## Benchmark Setup
- Start a EC2 instance according to the following table.

| Task      | Device  | EC2 Instance  |
|-----------|---------|---------------|
| Training  | GPU     | g4dn.4xlarge   |
| Inference | GPU     | g4dn.4xlarge   |
| Training  | Neuron  | trn1.2xlarge  |
| Inference | Neuron  | inf2.8xlarge  |

- Run [setup](setup/configure_neuron.sh)
- Build the docker image
- Run the image in interactive mode.
- Run the benchmark scripts.

```bash
torchrun --nproc_per_node=2 trainer_api_cli.py facebook/esm2_t6_8M_UR50D --device neuron
```

If want to skip the compilation before training:

[Follow this guide here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/training/pytorch-neuron-parallel-compile.html)

neuron_parallel_compile is an utility to extract graphs from trial run of your script, perform parallel compilation of the graphs, and populate the persistent cache with compiled graphs. Your trial run should be limited to about 100 steps, enough for the utility to extract the different graphs needed for full execution.
To avoid hang during extraction, please make sure to use xm.save instead of torch.save to save checkpoints.
After parallel compile, the actual run of your script will be faster since the compiled graphs are already cached. There may be additional compilations due to unreached execution paths, or changes in parameters such as number of data parallel workers.

```bash
neuron_parallel_compile torchrun --nproc_per_node=1 trainer_api.py facebook/esm2_t6_8M_UR50D --device neuron --epochs 0.1 --seed 42 --neuron-cache-url s3://nicolas-loka-bucket/neuron/esm2_t6_8M_UR50D
```

[Follow this guide here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide.html#neuron-compiler-cli-reference-guide-neuronx-cc )
```bash
neuronx-cc compile <model-files.hlo.pb> --framework XLA --target trn1 --model-type transformer --auto-cast none --optlevel 2 --output esm.neff --verbose info
```


## Benchmark Tasks
The benchmark comprises the following tasks:
1. Training using Protein Language Models.
2. Inference using Protein Language Models.

## Benchmark Results

### 1. Training Results

#### GPU
- **Training Time:** [Insert time taken for training on GPU]
- **Throughput:** [Insert throughput achieved during training on GPU]
- **Accuracy:** [Insert accuracy achieved during training on GPU]

#### AWS Neuron Device
- **Training Time:** [Insert time taken for training on AWS Neuron Device]
- **Throughput:** [Insert throughput achieved during training on AWS Neuron Device]
- **Accuracy:** [Insert accuracy achieved during training on AWS Neuron Device]

### 2. Inference Results

#### GPU
- **Inference Time:** [Insert time taken for inference on GPU]
- **Throughput:** [Insert throughput achieved during inference on GPU]

#### AWS Neuron Device
- **Inference Time:** [Insert time taken for inference on AWS Neuron Device]
- **Throughput:** [Insert throughput achieved during inference on AWS Neuron Device]

## Discussion
[Insert discussion of the benchmark results, including any insights gained from comparing performance on the two devices.]

## Conclusion
[Insert concluding remarks summarizing the performance of the deep learning framework on GPU and AWS Neuron Device for training and inference tasks using Protein Language Models.]

## Recommendations
[Insert any recommendations for optimizing performance on both devices based on the benchmark results.]

## Acknowledgments
[Insert any acknowledgments for contributors, resources, or funding related to the benchmarking process.]

## References
[Insert any references to relevant literature, tools, or methodologies used in conducting the benchmark.]
