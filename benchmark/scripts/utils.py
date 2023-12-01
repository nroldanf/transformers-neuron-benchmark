import os
import re
import boto3
from time import perf_counter
import torch
import requests
import numpy as np
from constants import WARM_UP_STEPS, INFERENCE_STEPS, DUMMY_INPUT

# TODO: Include batch size as a parameter. For now, batch is 1 and the sequence lenght changes


def generate_sample_inputs(tokenizer, max_sequence_length, batch_size=1, is_neuron=False):
    # TODO: adapt this for any model (pre-processing steps), OUTPUTS, device, etc
    inputs = [DUMMY_INPUT] * batch_size
    formatted = [format_protein_seqs(i) for i in inputs]
    tokenized = tokenizer(
        formatted, max_length=max_sequence_length, padding="max_length", return_tensors="pt"
    )
    if not is_neuron:
        tokenized.to("cuda")        
    return tuple(tokenized.values())


def measure_latency(model, tokenizer, sequence_length, batch_size, is_neuron=False):
    inputs = generate_sample_inputs(tokenizer, sequence_length, batch_size, is_neuron)
    # if is_neuron:
    #     import torch_neuronx
    #     model = torch_neuronx.dynamic_batch(model)
    
    print(inputs)
    latencies = []
    # warm up
    for _ in range(WARM_UP_STEPS):
        with torch.no_grad():
            _ = model(*inputs)
    # Timed run
    for _ in range(INFERENCE_STEPS):
        start_time = perf_counter()
        with torch.no_grad():
            _ = model(*inputs)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies, 95)
    return {
        "time_avg_ms": time_avg_ms,
        "time_std_ms": time_std_ms,
        "time_p95_ms": time_p95_ms,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
    }


def compile_model_inf1(model, tokenizer, max_sequence_length, batch_size, num_neuron_cores, is_neuron):
    os.environ["NEURON_RT_NUM_CORES"] = str(num_neuron_cores)
    import torch.neuron

    inputs = generate_sample_inputs(tokenizer, max_sequence_length, batch_size, is_neuron)
    return torch.neuron.trace(model, example_inputs=inputs)


def compile_model_inf2(model, tokenizer, max_sequence_length, batch_size, num_neuron_cores, is_neuron):
    # use only one neuron core
    os.environ["NEURON_RT_NUM_CORES"] = str(num_neuron_cores)
    import torch_neuronx

    inputs = generate_sample_inputs(tokenizer, max_sequence_length, batch_size, is_neuron)
    return torch_neuronx.trace(model, example_inputs=inputs)


def get_instance_type(region_name: str):
    # Create a Boto3 EC2 client
    ec2_client = boto3.client("ec2", region_name=region_name)

    # Get information about the current instance using the instance metadata service
    instance_id_url = "http://169.254.169.254/latest/meta-data/instance-id"
    instance_id_response = requests.get(instance_id_url)
    instance_id = instance_id_response.text

    # Describe the instance to get information, including the instance type
    response = ec2_client.describe_instances(InstanceIds=[instance_id])

    # Extract and return the instance type
    instance_type = response["Reservations"][0]["Instances"][0]["InstanceType"]
    return instance_type


def format_protein_seqs(protein_seq: str) -> str:
    """
    Formats a protein sequence by removing leading/trailing whitespace,
    replacing specific amino acids, and separating characters with spaces.

    Args:
        protein_seq (str): The input protein sequence.

    Returns:
        str: The formatted protein sequence.

    Example:
        >>> format_protein_seqs(" AUCGUOB ")
        'A U C G X X'
    """
    seq = protein_seq.strip()
    seq = re.sub(r"[UZOB]", "X", seq)
    seq = " ".join(seq)
    return seq