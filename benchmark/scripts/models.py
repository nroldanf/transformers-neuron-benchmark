from enum import Enum


class Device(str, Enum):
    "Define a device"

    gpu = "gpu"
    neuron = "neuron"


class PLMModel(str, Enum):
    "Define huggingface model id"

    esm1b = "facebook/esm1b_t33_650M_UR50S"
    esm2_8M = "facebook/esm2_t6_8M_UR50D"
    esm2_35M = "facebook/esm2_t12_35M_UR50D"
    esm2_150M = "facebook/esm2_t30_150M_UR50D"
    esm2_650M = "facebook/esm2_t33_650M_UR50D"
    esm2_3B = "facebook/esm2_t36_3B_UR50D"
    prot_bert = "Rostlab/prot_bert"
    prot_t5 = "Rostlab/prot_t5_xl_uniref50"
