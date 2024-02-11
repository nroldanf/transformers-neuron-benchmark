import os
import re
import boto3
from time import perf_counter
import torch
import requests
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from constants import *

# TODO: Include batch size as a parameter. For now, batch is 1 and the sequence lenght changes


def generate_sample_inputs(
    tokenizer, max_sequence_length, batch_size=1, is_neuron=False
):
    # TODO: adapt this for any model (pre-processing steps), OUTPUTS, device, etc
    inputs = [DUMMY_INPUT] * batch_size
    formatted = [format_protein_seqs(i) for i in inputs]
    tokenized = tokenizer(
        formatted,
        max_length=max_sequence_length,
        padding="max_length",
        return_tensors="pt",
    )
    if not is_neuron:
        tokenized.to("cuda")
    return tuple(tokenized.values())


def measure_latency(model, tokenizer, sequence_length, batch_size, is_neuron=False):
    inputs = generate_sample_inputs(tokenizer, sequence_length, batch_size, is_neuron)
    # enable the dynamic batch size
    # this only works for the non-zeroth dimension
    if is_neuron:
        import torch_neuronx

        model = torch_neuronx.dynamic_batch(model)

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


def compile_model_inf1(
    model, tokenizer, max_sequence_length, batch_size, num_neuron_cores, is_neuron
):
    os.environ["NEURON_RT_NUM_CORES"] = str(num_neuron_cores)
    import torch.neuron

    inputs = generate_sample_inputs(
        tokenizer, max_sequence_length, batch_size, is_neuron
    )
    return torch.neuron.trace(model, example_inputs=inputs)


def compile_model_inf2(
    model,
    tokenizer,
    max_sequence_length,
    batch_size,
    num_neuron_cores,
    compiler_args,
    compiler_workdir,
):
    # use only one neuron core
    os.environ["NEURON_RT_NUM_CORES"] = str(num_neuron_cores)
    import torch_neuronx

    inputs = generate_sample_inputs(tokenizer, max_sequence_length, batch_size, True)
    trace = torch_neuronx.trace(
        model,
        compiler_workdir=compiler_workdir,
        compiler_args=compiler_args,
        example_inputs=inputs,
    )
    return trace


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


def draw_pca(
    data: np.ndarray, n_components=2, title="", svd_solver="auto", random_state=42
):
    pca = PCA(
        n_components=n_components, svd_solver=svd_solver, random_state=random_state
    )
    u = pca.fit_transform(data)
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], range(len(u)))
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], u[:, 1])
    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], s=100)
    plt.title(title, fontsize=18)


def draw_tsne(data, n_components=2, random_state=42, perplexity=5, title=""):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        learning_rate="auto",
        n_iter=1500,
    )
    u = tsne.fit_transform(data)
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], range(len(u)))
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], u[:, 1])
    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], s=100)
    plt.title(title, fontsize=18)


def draw_umap(
    data, n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", title=""
):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
    )
    u = fit.fit_transform(data)
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], range(len(u)))
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], u[:, 1])
    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], s=100)
    plt.title(title, fontsize=18)
