# Fresh vs. Rotten Fruit Classification

This project utilizes deep learning to classify images of fruits as either fresh or rotten. It implements and compares three different Convolutional Neural Network (CNN) architectures for this task:

* **ResNet50**
* **AlexNet**
* **FruitNet (a custom CNN)**

The entire workflow, from data preprocessing and augmentation to model training and evaluation, is contained within the `frp.ipynb` Jupyter Notebook. The project is designed to be run in a Conda environment and leverages NVIDIA GPU acceleration for faster training.

## Dataset

The model is trained on a dataset of fresh and rotten apples, bananas, and oranges. The dataset contains a total of 14,747 images, split between the two categories for each fruit.

The dataset is downloaded from Google Drive and extracted automatically by running the second cell in the `frp.ipynb` notebook.

## Getting Started

### Prerequisites

To run this project, you will need a system with an NVIDIA GPU and Conda installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/vishalreddy2011/freshrotten_classification.git](https://github.com/vishalreddy2011/freshrotten_classification.git)
    cd freshrotten_classification
    ```
2.  **Create and activate the Conda environment:**
    The `env.yml` file contains all the necessary dependencies to run the notebook. Use the following command to create the environment:
    ```bash
    conda env create -f env.yml
    ```
    Then, activate the environment:
    ```bash
    conda activate tf-gpu
    ```

## Usage

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Open and run `frp.ipynb`:**
    Open the `frp.ipynb` notebook in Jupyter and run the cells sequentially to:
    * Download and extract the dataset.
    * Preprocess the data and apply image augmentation.
    * Train and evaluate the ResNet50, AlexNet, and FruitNet models.
    * View the classification reports and confusion matrices for each model.

## Models

### ResNet50

A pre-trained ResNet50 model is fine-tuned for the fruit classification task, achieving a final validation accuracy of **90.62%**.

### AlexNet

An implementation of the AlexNet architecture is trained from scratch. This model reaches a final validation accuracy of **94.83%**.

### FruitNet

This is a custom CNN architecture designed for this project. It consists of several convolutional and max-pooling layers, followed by dense layers for classification. FruitNet achieves a final validation accuracy of **94.97%**, slightly outperforming AlexNet.

## Results

All three models perform well on the task, with both AlexNet and the custom FruitNet achieving over 94% validation accuracy. The detailed classification reports and confusion matrices can be found in the output of the Jupyter Notebook.
````
