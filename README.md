# Robotic Grasp detection using ResNet34

## Requirements

* numpy
* opencv-python
* matplotlib
* scikit-image
* imageio
* torch
* torchvision
* torchsummary
* tensorboardX
* pyrealsense2
* Pillow

## Installation 

* Checkout the robotic grasping package

$ git clone https://git.uibk.ac.at/csaw8799/robot_grasp_detection

* Create a virtual environment

$ python3.6 -m venv --system-site-packages venv

* Source the virtual environment

$ source venv/bin/activate

* Install the requirements

$ cd robot_grasp_detection
$ pip install -r requirements.txt

## Dataset

This repository supports both the Cornell Grasping Dataset.

* Download Cornell Grasping Dataset

$ cd Dataset
$ bash get_cornell.sh

* Convert the PCD files to depth images by running 

$ python -m utils.dataset_processing.generate_cornell_depth Dataset

## Model Training

A model can be trained using the train_network.py script. Run train_network.py --help to see a full list of options.ù

python train_network.py --dataset cornell --dataset-path <Path To Dataset> --description training_cornell

