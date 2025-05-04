
# Plant Disease Detection

This project focuses on the detection of diseases in potato plants using image classification with deep learning. Two approaches were implemented: a custom Convolutional Neural Network (CNN) and a DenseNet121 model, both optimized with the Ant Colony Optimization (ACO) algorithm. [cite: 33, 45]

## Project Structure

The project directory is organized as follows:

* `data_dir`: Contains the dataset, including potato plant images categorized into 'Potato\_Early\_blight', 'Potato\_Late\_blight', and 'Potato\_healthy'. [cite: 28]
* `train_dir`: Subdirectory containing training images. [cite: 28]
* `val_dir`: Subdirectory containing validation images. [cite: 28]
* `potato_disease_classifier_aco.h5`: Saved model file for the custom CNN. [cite: 39]
* `potato_disease_classifier_aco_densenet.h5`: Saved model file for the DenseNet121 model. [cite: 49]
* `Untitled39.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, optimization, and evaluation. [cite: 28, 29, 33, 45]

## Dataset

The dataset consists of images of potato plants categorized into three classes:

* Potato Early blight
* Potato Late blight
* Potato healthy [cite: 28, 29]

The dataset was split into training and validation sets. [cite: 29]

## Models

### Custom CNN

A custom CNN model was developed using TensorFlow and Keras. The model architecture includes convolutional layers, max-pooling layers, flatten layers, dense layers, and dropout. The model parameters, such as the number of filters, kernel sizes, dense units, dropout rate, and learning rate, were optimized using the Ant Colony Optimization (ACO) algorithm. [cite: 33, 34]

### DenseNet121

The DenseNet121 model, pre-trained on ImageNet, was used as a base model. Additional layers were added for classification, and the ACO algorithm was used to optimize the hyperparameters. [cite: 45, 46]

## Optimization

The Ant Colony Optimization (ACO) algorithm was used to optimize the hyperparameters of both models. ACO is a metaheuristic algorithm inspired by the foraging behavior of ants. It was used to search for the best combination of parameters to maximize the validation accuracy of the models. [cite: 33, 46]

## Results

### Custom CNN

The custom CNN model achieved a validation accuracy of 87.5% [cite: 39]

### DenseNet121

The DenseNet121 model achieved a validation accuracy of 97.67% [cite: 52]

## Requirements

* tensorflow
* numpy
* scikit-learn
* matplotlib
* seaborn

## Installation

1.  Clone the repository.
2.  Download the dataset and place it in the `data_dir` directory.
3.  Install the required libraries using `pip install -r requirements.txt`.
4.  Run the Jupyter Notebook `Untitled39.ipynb` to train and evaluate the models. [cite: 28, 29, 33, 45]

## Usage

* To train the models, run the `Untitled39.ipynb` notebook.
* The trained models are saved as `potato_disease_classifier_aco.h5` and `potato_disease_classifier_aco_densenet.h5`. [cite: 39, 49]
* You can use the provided code to preprocess and predict diseases on new potato leaf images. [cite: 40, 41, 52]

## Contributing

Contributions to this project are welcome. Please submit a pull request with any proposed changes.

## License

This project is licensed under the \[License Name] License.

## Acknowledgements

* The dataset used in this project is publicly available.
* The TensorFlow and Keras libraries were used for building and training the models.
* The scikit-learn library was used for generating the classification report and confusion matrix. [cite: 42]
