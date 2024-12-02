# PyTorch TensorBoard Experiment: MNIST CNN Training

This project demonstrates how to integrate TensorBoard with PyTorch to visualize the training process of a simple convolutional neural network (CNN) on the MNIST dataset. The experiment explores the impact of different learning rates and batch sizes on the model's performance.

## Features

- Implementation of a CNN from scratch in PyTorch.
- Use of TensorBoard to:
  - Visualize model architecture.
  - Monitor training metrics like loss and accuracy.
  - Visualize input images and learned features.
  - Log hyperparameter tuning results.
- Dynamic hyperparameter tuning for learning rates and batch sizes.

## Requirements

To run this project, you'll need:

- Python 3.8 or higher
- PyTorch 1.9 or higher
- torchvision
- TensorBoard

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/matin-ghorbani/PyTorch-TensorBoard.git
    cd PyTorch-TensorBoard
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the training script:

    ```bash
    python pytorch_tensorboard.py
    ```

2. View TensorBoard logs:

    ```bash
    tensorboard --logdir=runs
    ```

3. Open the URL provided by TensorBoard in your browser (usually `http://localhost:6006`).

4. Monitor training progress, visualize the model architecture, and analyze hyperparameter effects.

## File Structure

- `pytorch_tensorboard.py`: Main script for training the model and logging data to TensorBoard.
- `data/`: Directory where the MNIST dataset will be downloaded.
- `runs/`: Directory where TensorBoard logs are saved.

## Hyperparameters

The training script evaluates the following hyperparameters:

- **Batch sizes**: `[32, 256]`
- **Learning rates**: `[1e-2, 1e-3, 1e-4, 1e-5]`

The results of each configuration are logged to TensorBoard for comparison.

## TensorBoard Visualizations

- **Model Graph**: Visualize the CNN structure.
- **Training Loss and Accuracy**: Track performance metrics across epochs.
- **Images and Features**: View input images and intermediate features.
- **Histograms**: Monitor weight distributions.
- **Embeddings**: Analyze learned feature representations.

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
