import typer
import logging
import os
import pandas as pd
import numpy as np
import torch
from torchinfo import summary
from datasets import Dataset, DatasetDict

# from sklearn.utils.class_weight import compute_class_weight
from models import Device, PLMModel
import tempfile
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    T5Tokenizer,
)
from sklearn.model_selection import train_test_split
import evaluate

app = typer.Typer(help="Benchmark Trainer API Script.")


def compute_metrics(eval_pred):
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics_results = clf_metrics.compute(predictions=predictions, references=labels)
    return metrics_results


def create_datasets(cytosolic_df: pd.DataFrame, membrane_df: pd.DataFrame):
    cytosolic_df["labels"] = 0
    membrane_df["labels"] = 1
    df = pd.concat([cytosolic_df, membrane_df], ignore_index=True)
    train_df, test_df = train_test_split(
        df, test_size=0.25, stratify=df["labels"], shuffle=True
    )
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    num_labels = (
        max(train_dataset["labels"] + test_dataset["labels"]) + 1
    )  # Add 1 since 0 can be a label
    return train_dataset, test_dataset, num_labels


def get_quant_config(quantization=None, dtype=torch.float16):
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["classifier"],
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_skip_modules=["classifier"]
        )
    else:
        return None


def get_model(
    model_name: str,
    mixed_precision: str = None,
    quantization=None,
    lora: bool = False,
    num_labels: int = 2,
    use_gradient_checkpointing: bool = False,
):
    datatype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name.value,
        quantization_config=get_quant_config(quantization, datatype),
        num_labels=num_labels,
    )
    summary(model)
    if lora:
        # https://huggingface.co/blog/AmelieSchreiber/esmbind
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            bias="none",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "query",
                "key",
                "value",
                # "EsmSelfOutput.dense",
                # "EsmIntermediate.dense",
                # "EsmOutput.dense",
                # "EsmContactPredictionHead.regression",
                # "EsmClassificationHead.dense",
                # "EsmClassificationHead.out_proj",
            ],
        )
        if quantization:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing)

        model = get_peft_model(model, peft_config)
        print("Model architecture after processing with PEFT:")
        summary(model)
        model.print_trainable_parameters()

    return model


@app.command()
def run(
    model_name: PLMModel,
    device: Device = Device.gpu,
    epochs: float = 1.0,
    seed: int = 42,
    neuron_cache_url: str = None,
    quantization=None,
    mixed_precision: str = None,
    lora: bool = False,
    use_gradient_checkpointing: bool = False,
    gradient_accumulation_steps: int = 1,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 2e-5,
    model_output_path: str = "models/",
):
    os.environ["NEURON_COMPILE_CACHE_URL"] = neuron_cache_url

    # Import correct package according to the device
    if device.value == "gpu":
        from transformers import TrainingArguments, Trainer

        # device = "gpu"
    elif device.value == "neuron":
        # from optimum.neuron import NeuronModelForSequenceClassification as AutoModelForSequenceClassification
        from optimum.neuron import NeuronTrainer as Trainer
        from optimum.neuron import NeuronTrainingArguments as TrainingArguments

        # device = "xla"

    # Load the data
    cytosolic_df = pd.read_parquet("../data/cytosolic.parquet")
    membrane_df = pd.read_parquet("../data/membrane.parquet")

    # Create the datasets
    train_dataset, val_dataset, num_labels = create_datasets(cytosolic_df, membrane_df)

    # Load the vocabulary
    if model_name == "Rostlab/prot_t5_xl_uniref50":
        tokenizer = T5Tokenizer.from_pretrained(model_name.value)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name.value)

    # Load the model
    model = get_model(
        model_name.value,
        mixed_precision,
        quantization,
        lora,
        num_labels,
        use_gradient_checkpointing,
    )
    # model.to(device)

    # Tokenize datasets
    def tokenize_function(dataset: DatasetDict):
        return tokenizer(dataset["Sequence"], padding="max_length", truncation=True)

    raw_datasets = DatasetDict({"train": train_dataset, "val": val_dataset})
    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=["Sequence"]
    )
    # Define the training arguments
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(
            output_dir=f"../models/{model_name.value.split('/')[-1]}-finetuned-localization",
            save_total_limit=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            seed=seed,
            gradient_checkpointing=use_gradient_checkpointing,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # group_by_length=True,  # Whether or not to group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient)
        )
        # Define the trainer
        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["val"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        if lora:
            trainer.model.save_pretrained(tmp_dir)
            # clear memory
            del model, trainer
            torch.cuda.empty_cache()
            # load PEFT model in fp16
            peft_config = PeftConfig.from_pretrained(tmp_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path,
                return_dict=True,
                torch_dtype=torch.float16,
                problem_type="single_label_classification",
            )
            model = PeftModel.from_pretrained(model, tmp_dir)
            model.eval()
            # Merge LoRA and base model and save
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(model_output_path)
        else:
            trainer.model.save_pretrained(model_output_path)


if __name__ == "__main__":
    app()
