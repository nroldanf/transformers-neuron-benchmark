REGION := us-east-1
# Login to neuron repos
# make ecr-login REGION=us-west-2 ECR_REPO=763104351884.dkr.ecr.us-west-2.amazonaws.com
ecr-login:
	aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPO}

build:
	echo "Building image ..."
	docker build -f ${DOCKERFILE} -t ${IMAGE_NAME} .

run-container-interactive-inf:
	docker run -it \
		-p 80:8000 \
		-v /home/ubuntu/workdir/transformers-neuron-benchmark/:/opt/app/inferentia \
		-e AWS_NEURON_VISIBLE_DEVICES=ALL \
		-e NEURON_RT_VISIBLE_CORES=0-1 ${IMAGE_NAME} bash

run-container-interactive-gpu:
	docker run -it \
		-p 80:8000 \
		--gpus=all \
		-v /home/ubuntu/workdir/transformers-neuron-benchmark/:/opt/app/inferentia \
		${IMAGE_NAME} bash
