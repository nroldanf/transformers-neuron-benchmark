# OCI runtime
sudo apt install -y golang && \
    export GOPATH=$HOME/go && \
    go get github.com/joeshaw/json-lossless && \
    cd /tmp/ && \
    git clone https://github.com/awslabs/oci-add-hooks && \
    cd /tmp/oci-add-hooks && \
    make build && \
    sudo cp /tmp/oci-add-hooks/oci-add-hooks /usr/local/bin/

# install OCI hook software

# INF1
# sudo apt-get install aws-neuron-runtime-base -y

# TRN1
# sudo apt-get install aws-neuronx-oci-hook -y

# For docker runtime
sudo cp /opt/aws/neuron/share/docker-daemon.json /etc/docker/daemon.json
sudo service docker restart
