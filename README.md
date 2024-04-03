# Deep Learning-Based Bacterial Colony Classification System

## Overview

Bacterial colony classification is a critical task in microbiology laboratories, facilitating the identification and differentiation of various bacterial species. Traditional methods rely heavily on manual observation and biochemical tests, which can be time-consuming and subjective. In recent years, the emergence of deep learning techniques has offered promising avenues for automating and improving this process.

This project focuses on developing a deep learning-based bacterial colony classification system using convolutional neural networks (CNNs). Specifically, the aim is to differentiate between Staphylococcus aureus and Escherichia coli colonies based on images captured from agar plates.

The proposed system leverages the power of PyTorch, a widely used deep learning framework, to construct a CNN architecture capable of learning discriminative features from bacterial colony images. By training the model on a dataset containing labeled images of Staphylococcus aureus and Escherichia coli colonies, we intend to create a robust classifier capable of accurately identifying these bacterial species.

This project holds significant potential for enhancing the efficiency and accuracy of bacterial colony classification in microbiology laboratories. By automating this process, we can reduce reliance on manual labor, minimize human error, and accelerate the pace of bacterial identification, ultimately contributing to advancements in healthcare, food safety, and environmental monitoring.

## Running the Code

1. **Clone Repository:**
```
git clone https://github.com/aedecano/Bact_colony_classifier_DL
```
2. **Navigate to Directory:**
```
cd Bact_colony_classifier_DL
```
3. **Run Script:**
```
python3 classify_colony_dl.py --train_data_path /path/to/train/data --val_data_path /path/to/validation/data --batch_size 32 --learning_rate 0.001 --epochs 10 --model_save_path /path/to/save/model.pth
```

**Note:**
- This script can benefit from GPU acceleration for faster training. Make sure you have access to a CUDA-enabled GPU if you wish to utilize GPU acceleration.
- If you don't have access to a GPU, you can still run the script on a CPU, but training may take significantly longer.

## Inspecting the Output Model

After training the model, you can inspect the saved model parameters by running:

```
python3 inspect_model.py --model_path /path/to/your/model.pth
```
