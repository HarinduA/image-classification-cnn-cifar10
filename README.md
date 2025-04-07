# Image Classification Using CNN on CIFAR-10 Dataset

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for **image classification** on the **CIFAR-10 dataset**. The CIFAR-10 dataset contains 60,000 32x32 color images categorized into 10 classes. The goal of this project is to classify images into one of these 10 classes with high accuracy using deep learning techniques.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Improvement Suggestions](#improvement-suggestions)
- [License](#license)

## Project Overview

This project applies deep learning techniques using **TensorFlow** to build and train a Convolutional Neural Network (CNN). The model is trained to classify CIFAR-10 images into one of the following categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The project is implemented in Python using TensorFlow and Keras, making it straightforward to run, modify, and experiment with different neural network architectures.

## Dataset

The **CIFAR-10 dataset** consists of:
- **60,000 32x32 color images**, divided into 10 categories.
- **10 classes** with 6,000 images per class.
- The classes are:
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

The dataset is already divided into **training** (50,000 images) and **test** (10,000 images) sets.

## Model Architecture

The Convolutional Neural Network (CNN) used in this project has the following architecture:

- **Convolutional Layers**: These layers apply convolutions to extract features such as edges, textures, and shapes in the images.
  - **Conv2D Layer** with 32 filters and a kernel size of (3x3).
  - **Conv2D Layer** with 64 filters and a kernel size of (3x3).
  - **Conv2D Layer** with 64 filters and a kernel size of (3x3).
  
- **MaxPooling Layers**: These layers reduce the spatial dimensions of the image, which helps to lower computational cost and reduces the chance of overfitting.
  - **MaxPooling2D Layer** with a pool size of (2x2).

- **Flatten Layer**: This layer converts the 2D matrices into a 1D vector to connect to the fully connected layers.

- **Fully Connected (Dense) Layers**:
  - **Dense Layer** with 64 units.
  - **Dense Layer** with 10 units (output layer), one for each class in CIFAR-10.

- **Activation Function**: ReLU activation is used for intermediate layers, while the output layer uses **softmax** activation (implicitly via the `SparseCategoricalCrossentropy` loss function) to output the probability of each class.

## Setup Instructions

### Prerequisites

Before running the project, make sure you have the following libraries installed:
- **Python 3.x**
- **TensorFlow**
- **Matplotlib**
- **NumPy**
- **Pandas**

To install the required dependencies, run the following command:

```bash
pip install tensorflow matplotlib numpy pandas
