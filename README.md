# Facial Expression Recognition (FER-2013) – Deep Learning with CNN

## 🧠 Project Overview
This project focuses on classifying human facial expressions into emotion categories using a Convolutional Neural Network (CNN). The model is trained on the FER-2013 dataset, which contains 48x48 grayscale face images labeled with 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## ⚙️ Technologies Used
- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- NumPy, Pandas, Matplotlib
- Image Augmentation

## 📊 Dataset
- **Source**: FER-2013 from Kaggle ([Link](https://www.kaggle.com/datasets/msambare/fer2013))
- 35,887 grayscale facial images (48x48 pixels)
- 7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## 📈 What I Did
- Preprocessed the dataset: normalization, reshaping, label encoding
- Built a CNN with Conv2D, MaxPooling, Dropout, and Dense layers
- Applied image augmentation to handle class imbalance
- Tuned hyperparameters (batch size, epochs, optimizer)
- Used Adam optimizer and categorical cross-entropy loss
- Visualized training curves and confusion matrix

## 🚀 Results
- Achieved **70.23% accuracy** on the test set
- Improved generalization using Dropout and Batch Normalization
- Used early stopping to prevent overfitting

## 📁 Files
- `fer_cnn_model.ipynb` – Full code in Jupyter notebook
- `README.md` – Project overview and documentation

## 🙋‍♂️ Author
Mirza Ansar Baig
