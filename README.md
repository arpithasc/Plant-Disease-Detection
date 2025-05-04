# Plant Disease Classification

This project uses a computer program to identify diseases in plant leaves. It can recognize different types of diseases and healthy plants.

## What's Inside

* `code.pdf`: The computer instructions for identifying diseases.
* `plant_dataset/`: The collection of plant leaf images used by the program.
* `plant_disease_classifier_aco_densenet.h5`: The trained program that can identify diseases.

## How It Works

1.  The program is designed to work on Google Colab (an online coding platform) and needs access to the plant images. [cite: 1]
2.  It organizes the images to learn and then check its accuracy. [cite: 2]
3.  The program learns what healthy and diseased leaves look like. [cite: 6, 7, 8, 9]
4.  It tests its knowledge on new images to see how well it can identify diseases. [cite: 15, 16, 17]
5.  The program can also predict the disease in a new plant leaf image you provide. [cite: 13, 14]

## What You'll See

The code will show how well the program performed using charts and numbers. [cite: 15, 16, 17]

## Future Ideas

Possible next steps include making the program more accurate or creating an easy way for anyone to use it.

## Code

The `code.pdf` file contains the Python scripts used in this project. This includes code for:
* Mounting Google Drive and accessing the dataset.
* Organizing and preprocessing the image data.
* Building and training a deep learning model (likely using a CNN architecture).
* Evaluating the model's performance using metrics like accuracy, classification report, and confusion matrix.
* Making predictions on sample images.


