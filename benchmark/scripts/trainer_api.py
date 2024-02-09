import typer
import logging
import pandas as pd
import numpy as np
from datasets import Dataset
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
    cytosolic_sequences = cytosolic_df["Sequence"].tolist()
    cytosolic_labels = [0 for protein in cytosolic_sequences]

    membrane_sequences = membrane_df["Sequence"].tolist()
    membrane_labels = [1 for protein in membrane_sequences]

    sequences = cytosolic_sequences + membrane_sequences
    labels = cytosolic_labels + membrane_labels

    assert len(sequences) == len(labels)

    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.25, shuffle=True
    )

    train_dataset = Dataset.from_pandas(train_sequences)
    test_dataset = Dataset.from_pandas(test_sequences)

    train_dataset = train_dataset.add_column("labels", train_labels)
    test_dataset = test_dataset.add_column("labels", test_labels)

    num_labels = max(train_labels + test_labels) + 1  # Add 1 since 0 can be a label

    return train_dataset, test_dataset, num_labels


@app.command()
def run(
    model_name: PLMModel,
    device: Device = Device.gpu,
    seed: int = 42,
):
    # Import correct package according to the device
    if device.value == "gpu":
        from transformers import TrainingArguments, Trainer
    elif device.value == "neuron":
        from optimum.neuron import NeuronTrainer as Trainer
        from optimum.neuron import NeuronTrainingArguments as TrainingArguments

    # Load the data
    cytosolic_df = pd.read_parquet("cytosolic.parquet")
    membrane_df = pd.read_parquet("membrane.parquet")

    # Create the datasets
    train_dataset, test_dataset, num_labels = create_datasets(cytosolic_df, membrane_df)

    # Load the vocabulary
    tokenizer = AutoTokenizer.from_pretrained(model_name.value)

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name.value, num_labels=num_labels
    )

    # Define the training arguments
    training_args = TrainingArguments(
        f"{model_name.value.split('/')[-1]}-finetuned-localization",
        output_dir="../models/",
        save_total_limit=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        seed=seed.value,
        group_by_length=True,  # Whether or not to group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient)
    )

    # Define the trainer
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    app()
