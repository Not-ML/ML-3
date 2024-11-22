#![pytorch-logo-dark](https://github.com/user-attachments/assets/3eb23528-872c-45c8-925a-ed656e757278)


# Advanced CNN for CIFAR-10 Classification

This project implements an advanced Convolutional Neural Network (CNN) in PyTorch to classify images from the CIFAR-10 dataset into 10 categories: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

## Features
- **Pretrained Dataset**: Utilizes CIFAR-10, a 60,000-image dataset split into 50,000 training and 10,000 test images.
- **Customizable CNN Model**: Includes multiple convolutional, batch normalization, pooling, and fully connected layers for accurate classification.
- **GPU Support**: Automatically utilizes CUDA if available for faster computation.
- **Data Augmentation**: Enhances training with random cropping, flipping, and normalization.
- **Training Scheduler**: Reduces learning rate dynamically for fine-tuning.

## Setup and Requirements
### Prerequisites
- Python 3.x
- Required packages: `torch`, `torchvision`, `tqdm`, `Pillow`

Install dependencies with:
```bash
pip install torch torchvision tqdm Pillow
```

### Clone the Repository
```bash
git clone https://github.com/your-username/advanced-cnn-cifar10.git
cd advanced-cnn-cifar10
```

## Training the Model
Run the script to train the model:
```bash
python advanced_cnn_cifar10.py
```
- Adjust `num_epochs`, `batch_size`, or `learning_rate` in the script for your requirements.
- Trained models are saved in the `checkpoints/` directory.

## Inference
### Predict Single Image
```python
from predict import predict_image

model_path = 'checkpoints/advanced_cnn.pth'
image_path = 'path/to/your/image.jpg'

prediction = predict_image(image_path, model_path)
print(f'Predicted class: {prediction}')
```

### Predict Batch of Images
```python
from predict import predict_batch

batch_predictions = predict_batch('path/to/dataset', model_path)
print(batch_predictions)
```

## Results
- Achieves 85-90% accuracy on the CIFAR-10 test set after 30 epochs.
- Checkpoint and accuracy logs are generated during training.

## Additional Notes
- Modify the architecture or hyperparameters to experiment with different configurations.
- GPU is highly recommended for faster training.

For more details, check the code files: `advanced_cnn_cifar10.py` (training) and `predict.py` (inference).
