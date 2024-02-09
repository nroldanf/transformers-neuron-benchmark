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

- ESM1-b from Meta AI
- ESM2 from Meta AI
- ProtBERT from RostLab
- ProtT5 from RostLab

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
