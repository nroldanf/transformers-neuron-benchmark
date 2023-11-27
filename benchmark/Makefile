run-all-gpu:
	echo "Running Rostlab/prot_t5_xl_half_uniref50-enc"
	python3 scripts/inference_transformers.py --model_id Rostlab/prot_t5_xl_half_uniref50-enc
	echo "Running Rostlab/prot_bert"
	python3 scripts/inference_transformers.py --model_id Rostlab/prot_bert
	echo "Running facebook/esm2_t36_3B_UR50D"
	python3 scripts/inference_transformers.py --model_id facebook/esm2_t36_3B_UR50D
	echo "Running facebook/esm2_t33_650M_UR50D"
	python3 scripts/inference_transformers.py --model_id facebook/esm2_t33_650M_UR50D
	echo "Running facebook/esm2_t30_150M_UR50D"
	python3 scripts/inference_transformers.py --model_id facebook/esm2_t30_150M_UR50D
	echo "Finished!"

run-all-inf:
	echo "Running Rostlab/prot_t5_xl_half_uniref50-enc"
	python3 scripts/inference_transformers.py --model_id Rostlab/prot_t5_xl_half_uniref50-enc --num_neuron_cores 2 --is_neuron=True
	echo "Running Rostlab/prot_bert"
	python3 scripts/inference_transformers.py --model_id Rostlab/prot_bert --num_neuron_cores 2 --is_neuron=True
	echo "Running facebook/esm2_t36_3B_UR50D"
	python3 scripts/inference_transformers.py --model_id facebook/esm2_t36_3B_UR50D --num_neuron_cores 2 --is_neuron=True
	echo "Running facebook/esm2_t33_650M_UR50D"
	python3 scripts/inference_transformers.py --model_id facebook/esm2_t33_650M_UR50D --num_neuron_cores 2 --is_neuron=True
	echo "Running facebook/esm2_t30_150M_UR50D"
	python3 scripts/inference_transformers.py --model_id facebook/esm2_t30_150M_UR50D --num_neuron_cores 2 --is_neuron=True
	echo "Finished!"
