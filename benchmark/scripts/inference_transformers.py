import logging
import argparse
import csv
from utils import (
    measure_latency,
    compile_model_inf1,
    compile_model_inf2,
    get_instance_type,
)
from constants import *

# TODO: Adapt this code to use typer for a better interface
# TODO: Make the batch dynamic in neuron so models are compiled only once

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_neuron", action="store_true")
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--sequence_length", type=int, default=None)
    # neuron specific args
    parser.add_argument("--num_neuron_cores", type=int, default=1)
    known_args, _ = parser.parse_known_args()
    return known_args


def main(args):
    logger.info(args)
    instance_type = get_instance_type(AWS_REGION)
    # TODO: Make the batch an array
    batch_size = 1

    # define sequence lengths to benchmark
    if args.sequence_length is None:
        sequence_lengths = SEQUENCE_LENGTHS
    else:
        sequence_lengths = [args.sequence_length]

    # benchmark model
    benchmark_dict = []

    logger.info(f"Running benchmark for model: {args.model_id}")
    logger.info(f"Batch size: {batch_size}")

    # load tokenizer and  model
    tokenizer_class, model_class = models.get(args.model_id)
    tokenizer = tokenizer_class.from_pretrained(args.model_id, legacy=False)
    model = model_class.from_pretrained(
        args.model_id, torchscript=True,
        torch_dtype=PRECISION,
        # quantization_config=quantization_config,
    )
    
    for sequence_length in sequence_lengths:
        # compile model if neuron
        if args.is_neuron:
            if "inf1" in instance_type:
                model = compile_model_inf1(
                    model, tokenizer, sequence_length, batch_size, args.num_neuron_cores
                )
            elif "inf2" in instance_type:
                model = compile_model_inf2(
                    model, tokenizer, sequence_length, batch_size, args.num_neuron_cores
                )
            else:
                raise ValueError("Unknown neuron version")
        else:
            model.to("cuda")

        logger.info(f"Measuring latency for sequence length {sequence_length}")
        res = measure_latency(model, tokenizer, sequence_length, batch_size, args.is_neuron)
        benchmark_dict.append({**res, "instance_type": instance_type})
    
    
    logger.info("Saving results...")
    # write results to csv
    keys = benchmark_dict[0].keys()
    output_file_name = f'results/benchmmark_{instance_type}_{args.model_id.replace("-","_").replace("/", "_")}.csv'
    with open(output_file_name, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(benchmark_dict)
    logger.info(f"Results saved to {output_file_name}")


if __name__ == "__main__":
    main(parse_args())
