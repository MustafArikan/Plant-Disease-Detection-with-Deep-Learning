Tomato Plant Disease Detection using Transfer Learning

Author: Mustafa Arıkan Date: December 22, 2025 

📌 Project Overview
Early diagnosis of diseases in agricultural production is critical for food safety and economic sustainability. This project presents a Deep Learning approach for the automatic detection of diseases (e.g., Early Blight, Bacterial Spot) in tomato plant leaves.

Using the MobileNetV2 architecture and Transfer Learning, the model is optimized to achieve high accuracy while reducing computational costs, offering an effective alternative to manual diagnosis methods.

📂 Dataset
The project utilizes the New Plant Diseases Dataset sourced from Kaggle.


Content: Images of tomato leaves captured under various lighting conditions, angles, and backgrounds.


Structure: The dataset consists of 10 classes (1 Healthy and 9 Diseased).


Data Split: 80% Training, 20% Validation.

Classes:
Tomato Bacterial Spot

Tomato Early Blight

Tomato Late Blight

Tomato Leaf Mold

Tomato Septoria Leaf Spot

Tomato Spider Mites (Two-spotted spider mite)

Tomato Target Spot

Tomato Yellow Leaf Curl Virus

Tomato Mosaic Virus

Tomato Healthy


🧠 Model Architecture & Methodology
To overcome data scarcity and shorten training time, a Transfer Learning approach was adopted rather than training a network from scratch.


Base Model: MobileNetV2 (pre-trained on ImageNet) was selected for its "Depthwise Separable Convolution" technology, which reduces parameter count and processing load, making it suitable for future mobile/Edge AI integration.

Preprocessing:

Resizing to 224×224 pixels.

Normalization to [0, 1] range.

One-Hot Encoding for categorical labels.

Custom Layers:

Feature Extractor: Frozen MobileNetV2 layers.


Global Average Pooling: To reduce feature maps to a 1D vector.


Dropout (0.2): To prevent overfitting.


Dense Output Layer: Softmax activation with neurons equal to the number of classes.

⚙️ Hyperparameters

Optimizer: Adam (Adaptive Moment Estimation).


Learning Rate: 0.0001.


Loss Function: Categorical Crossentropy.


Batch Size: 32.


Epochs: 10 (Monitored via Early Stopping).

📊 Results
The model demonstrated high stability and generalization capability during training.


Validation Accuracy: 86.2%.


Validation Loss: 0.41.

Analysis

Performance: The training and validation curves moved in parallel, indicating no overfitting issues.

Confusion Matrix: The model achieved near-perfect performance on distinct classes like Tomato Mosaic Virus and Yellow Leaf Curl Virus. Minor confusion was observed between visually similar diseases like Early Blight and Target Spot.


Test Confidence: On clear, focused test images, the model's confidence score reached 99-100%.

🛠️ Requirements
The project is implemented using Python and the TensorFlow/Keras framework.

Python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



🚀 Usage

Mount Drive (if using Colab): The code is designed to load data from Google Drive and copy it to a local Colab directory for speed .


Data Preparation: The ImageDataGenerator handles rescaling and augmentation (rotation, shifts, flips) for the training set .

Training: Run the training script. The model saves the best weights to best_tomato_model.h5.


Evaluation: The script generates accuracy/loss graphs and a confusion matrix heatmap.

🔮 Future Work
Future improvements aim to expand the model to include different plant species beyond tomatoes.

📄 References

Dataset: Vipul (vipoooool), New Plant Diseases Dataset, Kaggle.

Original Paper: Hughes, D., & Salathe, M. (2015). An open access repository of images on plant health....

Architecture: Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.
