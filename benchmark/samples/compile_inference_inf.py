import logging
import os
import re
import boto3
import requests
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    T5Tokenizer,
    T5Model,
    T5EncoderModel,
    EsmTokenizer,
    EsmModel,
    BertTokenizer,
    BertModel,
    BitsAndBytesConfig
)
# from optimum.neuron import NeuronModelForFeatureExtraction, pipeline

import torch

DUMMY_INPUT = "MNSTLT"

# Set up logging
logger = logging.getLogger(__name__)

models = {
    "Rostlab/prot_bert": (BertTokenizer, BertModel),
    "Rostlab/prot_t5_xl_half_uniref50-enc": (T5Tokenizer, T5EncoderModel),
    "Rostlab/prot_t5_xl_uniref50": (T5Tokenizer, T5Model),
    "facebook/esm2_t36_3B_UR50D": (EsmTokenizer, EsmModel),
    "facebook/esm2_t33_650M_UR50D": (EsmTokenizer, EsmModel),
    "facebook/esm2_t30_150M_UR50D": (EsmTokenizer, EsmModel),
}

model_id = "Rostlab/prot_t5_xl_half_uniref50-enc"
precision = torch.bfloat16
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     load_in_8bit=False,
#     bnb_4bit_compute_dtype=precision,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

def get_instance_type():
    # Create a Boto3 EC2 client
    ec2_client = boto3.client("ec2", region_name="us-east-1")

    # Get information about the current instance using the instance metadata service
    instance_id_url = "http://169.254.169.254/latest/meta-data/instance-id"
    instance_id_response = requests.get(instance_id_url)
    instance_id = instance_id_response.text

    # Describe the instance to get information, including the instance type
    response = ec2_client.describe_instances(InstanceIds=[instance_id])

    # Extract and return the instance type
    instance_type = response["Reservations"][0]["Instances"][0]["InstanceType"]
    return instance_type

def generate_sample_inputs(tokenizer, sequence_length, batch_size=1):
    # TODO: adapt this for any model (pre-processing steps), OUTPUTS, device, etc
    inputs = [DUMMY_INPUT] * batch_size
    inputs_formatted = [format_protein_seqs(i) for i in inputs]
    tokens = tokenizer(
        inputs_formatted, max_length=sequence_length, padding="max_length", return_tensors="pt"
    )
    return tuple(tokens.values())

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

def compile_model_inf2(model, tokenizer, sequence_length, batch_size, num_neuron_cores):
    # use only one neuron core
    os.environ["NEURON_RT_NUM_CORES"] = str(num_neuron_cores)
    import torch_neuronx

    payload = generate_sample_inputs(tokenizer, sequence_length, batch_size)
    return torch_neuronx.trace(model, payload)

def main():
    instance_type = get_instance_type()
    batch_size = 1
    num_neuron_cores = 2
    
    is_neuron = True

    # define sequence lengths to benchmark
    sequence_lengths = [16]
    

    for sequence_length in sequence_lengths:
        # load tokenizer and  model
        tokenizer_class, model_class = models.get(model_id)
        
        tokenizer = tokenizer_class.from_pretrained(model_id, legacy=False)
        model = model_class.from_pretrained(
            model_id, torchscript=True,
            torch_dtype=precision,
            # quantization_config=quantization_config,
        )

    # compile model if neuron
    if is_neuron:
        if "inf1" in instance_type:
            model = compile_model_inf1(
                model, tokenizer, sequence_length, batch_size, num_neuron_cores
            )
        elif "inf2" in instance_type:
            model = compile_model_inf2(
                model, tokenizer, sequence_length, batch_size, num_neuron_cores
            )
        else:
            raise ValueError("Unknown neuron version")
    else:
        model.to("cuda")

    inputs = generate_sample_inputs(tokenizer, sequence_length, batch_size)
    with torch.no_grad():
        _ = model(*inputs)

if __name__ == "__main__":
    main()
