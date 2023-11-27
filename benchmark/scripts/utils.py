import os
import boto3
import requests
from time import perf_counter
import numpy as np
from constants import WARM_UP_STEPS, INFERENCE_STEPS, DUMMY_INPUT

# TODO: Include batch size as a parameter. For now, batch is 1 and the sequence lenght changes


def generate_sample_inputs(tokenizer, sequence_length, batch_size=1):
    # TODO: adapt this for any model (pre-processing steps), OUTPUTS, device, etc
    inputs = [DUMMY_INPUT] * batch_size
    embeddings = tokenizer(
        inputs, max_length=sequence_length, padding="max_length", return_tensors="pt"
    )
    return tuple(embeddings.values())


def measure_latency(model, tokenizer, sequence_length, batch_size):
    payload = generate_sample_inputs(tokenizer, sequence_length, batch_size)
    latencies = []
    # warm up
    for _ in range(WARM_UP_STEPS):
        _ = model(*payload)
    # Timed run
    for _ in range(INFERENCE_STEPS):
        start_time = perf_counter()
        _ = model(*payload)
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


def compile_model_inf1(model, tokenizer, sequence_length, batch_size, num_neuron_cores):
    os.environ["NEURON_RT_NUM_CORES"] = str(num_neuron_cores)
    import torch.neuron

    payload = generate_sample_inputs(tokenizer, sequence_length, batch_size)
    return torch.neuron.trace(model, payload)


def compile_model_inf2(model, tokenizer, sequence_length, batch_size, num_neuron_cores):
    # use only one neuron core
    os.environ["NEURON_RT_NUM_CORES"] = str(num_neuron_cores)
    import torch_neuronx

    payload = generate_sample_inputs(tokenizer, sequence_length, batch_size)
    return torch_neuronx.trace(model, payload)


def get_instance_type():
    # Create a Boto3 EC2 client
    ec2_client = boto3.client("ec2")

    # Get information about the current instance using the instance metadata service
    instance_id_url = "http://169.254.169.254/latest/meta-data/instance-id"
    instance_id_response = requests.get(instance_id_url)
    instance_id = instance_id_response.text

    # Describe the instance to get information, including the instance type
    response = ec2_client.describe_instances(InstanceIds=[instance_id])

    # Extract and return the instance type
    instance_type = response["Reservations"][0]["Instances"][0]["InstanceType"]
    return instance_type
