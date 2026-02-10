# Plant Disease Detection with Deep Learning

## Project Overview
This project focuses on multi-class classification of tomato leaf diseases using a deep learning approach based on transfer learning. The system analyzes plant leaf images and predicts disease categories to support early detection and decision-making in agriculture.

A pretrained convolutional neural network was used as a feature extractor, and a custom classification head was trained on a tomato leaf dataset containing multiple disease categories along with healthy samples.

The project demonstrates the full deep learning pipeline including preprocessing, model training, evaluation, visualization, and performance analysis.

---

## Objectives
- Perform multi-class image classification on plant leaf images
- Detect various tomato diseases automatically
- Apply transfer learning using a pretrained CNN
- Evaluate model performance using multiple evaluation metrics
- Visualize training progress and classification performance

---

## Dataset
PlantVillage Tomato Leaf Dataset

Classes:
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Spider Mites
- Tomato Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Tomato Healthy

Dataset Characteristics:
- Image-based multi-class classification problem
- Balanced representation across disease categories
- RGB images resized before training

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

---

## Model Architecture
- Base Model: MobileNetV2 (Transfer Learning)
- Input Size: 224x224 RGB images
- Feature Extraction using pretrained ImageNet weights
- Custom Dense Classification Layers
- Activation Function: ReLU / Softmax (output layer)
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Evaluation Metric: Accuracy

---

## Training Strategy
- Transfer learning with pretrained convolutional layers
- Initial layers frozen during early training
- Custom classification head trained on tomato dataset
- Image resizing and normalization applied
- Training and validation split used
- Monitoring training and validation loss to control overfitting

---

## Training and Validation Performance

![Training and Validation Results](Graphics/PDD0.png)

The training and validation curves show stable convergence. Validation accuracy follows training accuracy closely, indicating controlled overfitting and consistent learning behaviour throughout epochs.

---

## Confusion Matrix

![Confusion Matrix](Graphics/PDD1.png)

The confusion matrix highlights class-wise prediction performance.

Observations:
- High accuracy in Healthy and Viral disease classes
- Some confusion between visually similar diseases such as Early Blight and Late Blight
- Spider Mites and Target Spot occasionally overlap due to similar texture patterns
- Most predictions remain concentrated along the diagonal, indicating strong classification performance

---

## Evaluation Metrics
- Accuracy
- Confusion Matrix
- Class-wise prediction distribution
- Training Loss and Validation Loss Monitoring

Model Performance Summary:
- Validation Accuracy: approximately 85%
- Stable training process
- Minimal divergence between training and validation curves
- Strong performance across most disease categories

---

## Project Structure
Plant-Disease-Detection-with-Deep-Learning/
│
├── Graphics/
│ ├── PDD0.png
│ └── PDD1.png
│
├── Plant_Disease_Detection.ipynb
└── README.md

---

## Installation

Clone the repository: 
```bash
git clone https://github.com/MustafArikan/Plant-Disease-Detection-with-Deep-Learning.git
cd Plant-Disease-Detection-with-Deep-Learning
```

---

## Requirements

- Python 3.9+
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

GPU is optional but recommended for faster training.

---

## Reproducibility

To reproduce the results:

1. Prepare the dataset in the correct directory structure
2. Open the notebook file
3. Run all cells sequentially
4. Monitor training graphs and evaluation outputs

---

## Future Improvements

- Advanced data augmentation
- Hyperparameter optimization
- Fine-tuning deeper layers
- Testing EfficientNet and ResNet architectures
- Real-time deployment for plant disease detection

---

## Author
Mustafa Arıkan


