import os
import logging
import random
import math
import gc
import string
import numpy as np
import pandas as pd
import argparse
import git
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    # TrainingArguments,
    # Trainer,
    DataCollatorWithPadding,
)
from transformers.integrations import MLflowCallback
import mlflow
from datasets import Dataset, DatasetDict
import evaluate

# from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronTrainer as Trainer
from optimum.neuron import NeuronTrainingArguments as TrainingArguments

# from optimum.neuron.distributed import lazy_load_for_parallelism

# MLFlow Tracking server URL
# TODO: Make this URI a secret or environment variable
tracking_uri = os.getenv("MLFLOW_SERVER_URI")
mlflow.set_tracking_uri(tracking_uri)
mlflow.get_tracking_uri()  # This checks if it was set properly
# Whether to use MLflow .log_artifact() facility to log artifacts. Will copy whatever is in TrainingArguments’s output_dir to the local or remote artifact storage
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "TRUE"


def get_git_hash():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.commit.hexsha
    short_sha = repo.git.rev_parse(sha, short=7)
    return short_sha


def compute_metrics(eval_pred):
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics_results = clf_metrics.compute(predictions=predictions, references=labels)
    return metrics_results


def train(args, raw_datasets):
    # Download model from model hub
    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.hidden_dropout_prob = args.hidden_dropout_prob
    model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    model_config.classifier_dropout = args.classifier_dropout
    model_config.layer_norm_eps = args.layer_norm_eps

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=model_config
    )
    # Freeze encoder (attention heads and linear layers) weights
    # reuse the lower layers and retrain the upper ones if the tasks differ from the classification
    if args.freeze_encoder:
        layers_to_freeze = [
            f"encoder.layer.{i}." for i in range(0, args.lower_layers_to_freeze)
        ]
        for name, param in model.base_model.named_parameters():
            for layer in layers_to_freeze:
                if name.startswith(layer):
                    # Don't update this weights using gradient descent
                    param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_function(dataset: DatasetDict):
        # Truncate if exceed maximum length (otherwise it will throw an error)
        # Max length: 512 tokens for BERT
        return tokenizer(dataset["text"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    logger.info(f"Loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f"Loaded val_dataset length is: {len(val_dataset)}")

    experiment_name = f"transformers-{args.model_name}"
    mlflow.set_experiment(experiment_name)

    random_string = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=12)
    )

    # lr_num_training_steps = math.floor((len(train_dataset) / args.train_batch_size)) * args.epochs
    # logger.info(f"lr training steps (all epochs): {lr_num_training_steps}")

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        save_total_limit=1,
        logging_first_step=True,
        overwrite_output_dir=True,
        report_to=["mlflow"],
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        lr_scheduler_type=args.scheduler_type,
        warmup_steps=args.warmup_steps,
        # num_training_steps=num_training_steps,
        seed=42,
        fp16=args.fp16,
        run_name=random_string,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_dir=f"{args.output_data_dir}/logs",
        save_steps=500,  # Checkpoint callback frequency
        logging_steps=20,  # Number of update steps between two logs if `logging_strategy="steps"
        eval_steps=20,
        group_by_length=True,  # Whether or not to group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient)
    )
    # set optimizer
    training_args = training_args.set_optimizer(
        name=args.optimizer_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta_1,
        beta2=args.beta_2,
        epsilon=args.epsilon,
    )
    # TODO: Change train_dataset and test_dataset (torch.utils.data.IterableDataset/Dataset) to use s3 torch connector for s3
    # to not download the data to local directory first
    trainer = Trainer(
        model=model,
        args=training_args,
        # callbacks=[MLflowCallback()], # Callbacks
        compute_metrics=compute_metrics,  # Function that will be used to compute metrics at evaluation
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
        # data_collator=data_collator,
    )

    # Log params from preprocessing pipeline
    mlflow.log_params(preprocessing_dict)
    # Log dataset inputs
    mlflow.log_input(dataset=mlflow_dataset_train, context="training")
    mlflow.log_input(dataset=mlflow_dataset_val, context="validation")
    # Log git hash tag
    mlflow.set_tag("mlflow.source.git.commit", get_git_hash())

    # Train model
    trainer.train(resume_from_checkpoint=False)

    # Save model to s3
    logger.debug(f"saving model to {args.model_dir}")
    trainer.save_model(args.model_dir)

    mlflow.end_run()

    logger.info("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--spelling_opt", type=str, default="autocorrect")
    parser.add_argument("--preprocessing_steps", type=str, default=None)
    parser.add_argument("--preprocessing_steps_values", type=str, default=None)
    # optimizer hyperparams
    parser.add_argument("--scheduler_type", type=str, default="polynomial")
    parser.add_argument("--optimizer_name", type=str, default="adamw_torch")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    # regularization params
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--classifier_dropout", type=float, default=0.1)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    # layers to freeze in the encoder
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--lower_layers_to_freeze", type=int, default=8)
    # Data, model, and output directories
    parser.add_argument(
        "--output_data_dir", type=str, default=os.getenv("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument("--model_dir", type=str, default=os.getenv("SM_MODEL_DIR"))
    parser.add_argument(
        "--training_dir", type=str, default=os.getenv("SM_CHANNEL_TRAIN")
    )
    parser.add_argument("--val_dir", type=str, default=os.getenv("SM_CHANNEL_TEST"))

    args, _ = parser.parse_known_args()
    # Parse the preprocessing steps
    preprocessing_steps_keys_split = args.preprocessing_steps.split("-")
    preprocessing_steps_values_split = args.preprocessing_steps_values.split("-")
    preprocessing_steps_values_bool = [
        bool(entry == "True") for entry in preprocessing_steps_values_split
    ]
    preprocessing_dict = dict(
        zip(preprocessing_steps_keys_split, preprocessing_steps_values_bool)
    )
    preprocessing_dict["spelling_opt"] = args.spelling_opt

    # Set up logging
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger.info(args.training_dir)
    logger.info(args.val_dir)

    # cleaned_message, message, label
    train_dataset_df = pd.read_parquet(f"{args.training_dir}/train_dataset.parquet")
    val_dataset_df = pd.read_parquet(f"{args.val_dir}/val_dataset.parquet")
    # Define mlflow datasets
    mlflow_dataset_train = mlflow.data.from_pandas(
        train_dataset_df.copy(deep=False),
        source=f"{args.training_dir}/train_dataset.parquet",
    )
    mlflow_dataset_val = mlflow.data.from_pandas(
        val_dataset_df.copy(deep=False), source=f"{args.val_dir}/val_dataset.parquet"
    )

    train_dataset_df.drop(columns=["message"], inplace=True)
    val_dataset_df.drop(columns=["message"], inplace=True)

    train_dataset_df.rename(columns={"cleaned_message": "text"}, inplace=True)
    val_dataset_df.rename(columns={"cleaned_message": "text"}, inplace=True)

    logger.debug(train_dataset_df.columns)
    logger.debug(val_dataset_df.columns)

    train_dataset = Dataset.from_pandas(train_dataset_df)
    val_dataset = Dataset.from_pandas(val_dataset_df)

    del train_dataset_df, val_dataset_df
    gc.collect()

    # Tokenizing datasets
    raw_datasets = DatasetDict({"train": train_dataset, "val": val_dataset})

    train(args, raw_datasets)
