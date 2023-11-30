import torch
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

WARM_UP_STEPS = 10
INFERENCE_STEPS = 100
DUMMY_INPUT = "MNSTLT"
AWS_REGION = "us-west-2"
PRECISION = torch.bfloat16
SEQUENCE_LENGTHS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

models = {
    "Rostlab/prot_bert": (BertTokenizer, BertModel),
    "Rostlab/prot_t5_xl_half_uniref50-enc": (T5Tokenizer, T5EncoderModel),
    "Rostlab/prot_t5_xl_uniref50": (T5Tokenizer, T5Model),
    "facebook/esm2_t36_3B_UR50D": (EsmTokenizer, EsmModel),
    "facebook/esm2_t33_650M_UR50D": (EsmTokenizer, EsmModel),
    "facebook/esm2_t30_150M_UR50D": (EsmTokenizer, EsmModel),
}

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=PRECISION,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)