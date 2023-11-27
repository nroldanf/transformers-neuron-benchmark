# Update OS packages
sudo apt-get update -y

# Update OS headers
sudo apt-get install linux-headers-$(uname -r) -y

# Install git
sudo apt-get install git -y

# ****************************************************************
# NEURON CONFIGURATION
# ****************************************************************

# update Neuron Driver
sudo apt-get update aws-neuronx-dkms=2.* -y

# Update Neuron Runtime
# Runtime Library consists of the libnrt.so and header files
sudo apt-get install aws-neuronx-collectives=2.* -y
sudo apt-get install aws-neuronx-runtime-lib=2.* -y

# Update Neuron Tools
sudo apt-get install aws-neuronx-tools=2.* -y
# ****************************************************************

# Add PATH
export PATH=/opt/aws/neuron/bin:$PATH
