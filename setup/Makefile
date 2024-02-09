REGION := us-east-1

# Login to neuron repos
# make ecr-login REGION=us-west-2 ECR_REPO=763104351884.dkr.ecr.us-west-2.amazonaws.com
ecr-login:
	aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPO}

# Build inferentia2
# make build DOCKERFILE=Dockerfile.inf2 IMAGE_NAME=transformers-inf2
# Build GPU
# make build DOCKERFILE=Dockerfile.gpu IMAGE_NAME=transformers-gpu
build:
	echo "Building image ..."
	docker build -f ${DOCKERFILE} -t ${IMAGE_NAME} .

# Run inferentia container
# make run-container-interactive-inf IMAGE_NAME=transformers-inf2
run-container-interactive-inf:
	docker run -it \
		-p 80:8000 \
		-v /home/ubuntu/workdir/transformers-neuron-benchmark/:/opt/app/inferentia \
		-e AWS_NEURON_VISIBLE_DEVICES=ALL -e NEURON_RT_VISIBLE_CORES=0-5 -e NEURON_RT_NUM_CORES=0-5 ${IMAGE_NAME} bash

# Run trainium container
# make run-container-interactive-trn IMAGE_NAME=neuron-trn1
run-container-interactive-trn:
	docker run -it \
		-v /home/ubuntu/workdir/transformers-neuron-benchmark/:/opt/app/trainium \
		-e AWS_NEURON_VISIBLE_DEVICES=ALL ${IMAGE_NAME} bash


# Run gpu container
# make run-container-interactive-gpu IMAGE_NAME=transformers-gpu
run-container-interactive-gpu:
	docker run -it \
		-p 80:8000 \
		--gpus=all \
		-v /home/ubuntu/workdir/transformers-neuron-benchmark/:/opt/app \
		${IMAGE_NAME} bash
