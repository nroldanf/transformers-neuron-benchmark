import typer
import logging
import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from models import Device, PLMModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
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


@app.command()
def run(
    model_name: PLMModel,
    device: Device = Device.gpu,
    epochs: float = 1.0,
    seed: int = 42,
    neuron_cache_url: str = None,
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
    tokenizer = AutoTokenizer.from_pretrained(model_name.value)

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name.value, num_labels=num_labels
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
    training_args = TrainingArguments(
        output_dir=f"../models/{model_name.value.split('/')[-1]}-finetuned-localization",
        save_total_limit=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        seed=seed,
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


if __name__ == "__main__":
    app()
