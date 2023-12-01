import torch
from esm import FastaBatchedDataset, pretrained
from esm.model.esm2 import ESM2


if __name__ == "__main__":
    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ]

    print(f"Inputs lengths: {[len(i[1]) for i in data]}")    

    model_name = "esm2_t33_650M_UR50D"
    
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()  # disables dropout for deterministic results

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        
    print(f"Logits shape: {results['logits'].shape}")    
    print(f"Representations shape: {results['representations'][33].shape}")
    
    token_representations = results["representations"][33]
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    print(f"Per sequence representation shape: {[i.shape for i in sequence_representations]}")