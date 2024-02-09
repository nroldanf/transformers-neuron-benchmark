import logging
import torch
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
    EsmForMaskedLM,
    BertTokenizer,
    BertModel,
    BitsAndBytesConfig,
)


if __name__ == "__main__":
    model_id = "facebook/esm2_t6_8M_UR50D"
    # precision = torch.bfloat16()

    sequences = ["MNSTLTVFLKEVRENIRDRRTVINTLFVGPLMAPLIFVLLINTLVTRELSKAEKPLPLPV"]
    input_len = [len(seq.replace(" ", "")) for seq in sequences]

    print([len(i) for i in sequences])

    tokenizer = EsmTokenizer.from_pretrained(model_id)
    model = EsmModel.from_pretrained(model_id, torchscript=True)
    model.requires_grad_(False)  # freeze weights
    model.eval()
    model.to("cuda")

    tokenized = tokenizer(
        sequences, add_special_tokens=False, padding=True, return_tensors="pt"
    )
    tokenized.to("cuda")

    with torch.no_grad():
        outputs = model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )

    representations = outputs.last_hidden_state.mean(dim=1)

    print(representations.shape)
