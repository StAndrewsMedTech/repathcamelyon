

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = repathcamelyon
PYTHON_INTERPRETER = python
PYTHON_VERSION = 3.6

## test if Anaconda is installed
ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

# network
JUPYTER_PORT := 8881

#################################################################################
# PYTHON ENVIRONMENT COMMANDS                                                   #
#################################################################################

## set up the python environment
create_environment:
ifeq (True,$(HAS_CONDA))
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
	@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@echo "Conda is not installed. Please install it."
endif

## install the requirements into the python environment
requirements: install_curl install_asap install_openslide
	conda env update --file environment.yml
	pip install -r requirements.txt

## save the python environment so it can be recreated
export_environment:
	conda env export --no-builds | grep -v "^prefix: " > environment.yml
	# note - the requirements.txt. is required to build the
	# environment up but is not changed are part of the export
	# process

# some packages that are required by the project have binary dependencies that
# have to be installed out with Conda.

## download and install ASAP
ASAP_LOCATION = https://github.com/computationalpathologygroup/ASAP/releases/download/1.9/ASAP-1.9-Linux-Ubuntu1804.deb
install_asap:
	sudo apt-get update
	sudo curl -o ASAP-1.9-Linux-Ubuntu1804.deb -L $(ASAP_LOCATION)
	sudo apt -y install ./ASAP-1.9-Linux-Ubuntu1804.deb
	sudo rm ASAP-1.9-Linux-Ubuntu1804.deb

install_openslide:
	sudo apt-get update
	sudo apt install -y build-essential
	sudo apt-get -y install openslide-tools
	pip install Pillow
	pip install openslide-python

install_curl:
	sudo apt -y install curl


#################################################################################
# CONTAINER COMMANDS                                                            #
#################################################################################
docker_image:
	docker build -t $(PROJECT_NAME) .

docker_run:
	docker run --shm-size=16G \
				--gpus all -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
				-v $(PROJECT_DIR):/home/ubuntu/$(PROJECT_NAME) \
				-v /raid/datasets:/home/ubuntu/$(PROJECT_NAME)/data \
				-v /raid/experiments/$(PROJECT_NAME):/home/ubuntu/$(PROJECT_NAME)/experiments \
				-v /mnt/isilon1/camelyon17:/home/ubuntu/$(PROJECT_NAME)/data/camelyon17 \
				-it $(PROJECT_NAME):latest

docker_run_local:
	docker run --shm-size=16G --gpus all -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
				-v $(PROJECT_DIR):/home/ubuntu/$(PROJECT_NAME) \
				-it $(PROJECT_NAME):latest

#################################################################################
# JUPYTER COMMANDS                                                              #
#################################################################################
setup_jupyter:
	pip install --user ipykernel
	python -m ipykernel install --user --name=$(PROJECT_NAME)

run_notebook:
	jupyter lab --ip=* --port $(JUPYTER_PORT) --allow-root

run_lab:
	jupyter lab --ip=* --port $(JUPYTER_PORT) --allow-root

run_lab_tb:
	jupyter lab --ip=* --port $(JUPYTER_PORT) --allow-root

#################################################################################
# DATA PRE-PROCESSING COMMANDS                                                  #
#################################################################################
