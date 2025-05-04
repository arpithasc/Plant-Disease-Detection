from google.colab import drive

drive.mount('/content/drive', force_remount=True)

import os

data_dir = '/content/drive/MyDrive/potato_dataset'

categories = os.listdir(data_dir)

categories = sorted(categories)

print(categories)

import zipfile

def unzip_file(zip_path, extract_to):
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# Define paths

base_dir = '/content/drive/MyDrive/potato_dataset'

train_dir = os.path.join(base_dir, 'train')

val_dir = os.path.join(base_dir, 'validation')

#Unzip files

zip_files = [
    'Potato_Early_blight.zip',
    'Potato_healthy.zip',
    'Potato_Late_blight.zip'
]

#Unzip the main dataset files
for zip_file in zip_files:
  unzip_file(os.path.join(base_dir, zip_file), base_dir)

#Unzip the split dataset files if they exist
for zip_file in zip_files:
  train_zip = os.path.join(train_dir, zip_file)
  val_zip = os.path.join(val_dir, zip_file)
  if os.path.exists(train_zip):
    unzip_file(train_zip, train_dir)
  if os.path.exists(val_zip):
    unzip_file(val_zip, val_dir)

import shutil
import random

def organize_dataset(base_dir, train_dir, val_dir):
  classes = ['Potato_Early_blight', 'Potato_healthy', 'Potato _Late_blight']

  for cls in classes:
    cls_dir = os.path.join(base_dir, cls)
    if os.path.exists(cls_dir):
      images = os.listdir(cls_dir)
      random.shuffle(images)
      val_count = int(len(images) * 0.2)
      train_images = images[val_count:]
      val_images = images[:val_count]

      # Move images to train directory
      train_cls_dir = os.path.join(train_dir, cls)
      os.makedirs(train_cls_dir, exist_ok=True)
      for img in train_images:
        shutil.move(os.path.join(cls_dir, img), os.path.join(train_cls_dir, img))

      #Move images to validation directory
      val_cls_dir = os.path.join(val_dir, cls)
      os.makedirs(val_cls_dir, exist_ok=True)
      for img in val_images:
        shutil.move(os.path.join(cls_dir, img), os.path.join(val_cls_dir, img))

# Organize the dataset
organize_dataset(base_dir, train_dir, val_dir)

def verify_dataset_structure(directory):
  for dirpath, dirnames, filenames in os.walk(directory):
    print(f'Found directory: {dirpath}')
    for file_name in filenames:
      print(f'\t{file_name}')

print("Train directory structure:")
verify_dataset_structure(train_dir)

print("Validation directory structure:")
verify_dataset_structure(val_dir)

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

train_dir = '/content/drive/MyDrive/potato_dataset/train'
val_dir = '/content/drive/MyDrive/potato_dataset/validation'

img_height = 150
img_width = 150
batch_size = 32

# Define the CNN model
def create_model(params):
  model = Sequential([
      Conv2D(params['filters1'], (params['kernel_size1'], params['kernel_size1']), activation='relu', input_shape=(img_height, img_width, 3)),
      MaxPooling2D((2, 2)),
      Conv2D(params['filters2'], (params['kernel_size2'], params['kernel_size2']), activation='relu'),
      MaxPooling2D((2, 2)),
      Flatten(),
      Dense(params['dense_units'], activation='relu'),
      Dropout(params['dropout']),
      Dense(3, activation='softmax')
  ])
  model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

# Define ACO parameters
class Ant:
  def __init__(self):
    self.position = None
    self.cost = float('inf')

def random_params():
  return {
      'filters1': np.random.choice([32, 64, 128]),
      'kernel_size1': np.random.choice([3, 5]),
      'filters2': np.random.choice([32, 64, 128]),
      'kernel_size2': np.random.choice([3, 5]),
      'dense_units': np.random.choice([128, 256, 512]),
      'dropout': np.random.uniform(0.2, 0.5),
      'learning_rate': np.random.choice([1e-3, 1e-4, 1e-5])
  }

def aco_optimize(num_ants, num_generations):
  best_ant = Ant()

  for generation in range(num_generations):
    ants = [Ant() for _ in range(num_ants)]

    for ant in ants:
      ant.position = random_params()
      model = create_model(ant.position)
      history = model.fit(train_generator, epochs=5, validation_data=validation_generator, verbose=0)
      val_accuracy = history.history['val_accuracy'][-1]
      ant.cost = -val_accuracy

      if ant.cost < best_ant.cost:
        best_ant.position = ant.position
        best_ant.cost = ant.cost

    print(f'Generation {generation + 1}, Best Cost: {-best_ant.cost}')

  return best_ant.position

# Data augmentation and data generators

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Optimize

best_params = aco_optimize(num_ants=10, num_generations=3)
print('Best Parameters:', best_params)

# Evaluate the optimized model

best_model = create_model(best_params)
history = best_model.fit(train_generator, epochs=20, validation_data=validation_generator, verbose=1)

# Save the model

best_model.save('potato_disease_classifier_aco.h5')

#Evaluate the model

val_loss, val_accuracy = best_model.evaluate(validation_generator)
print(f'Validation loss: {val_loss}')
print(f'Validation accuracy: {val_accuracy}')

import os

# Define the path to your directory

directory_path = '/content/drive/MyDrive/potato_dataset/train/Potato_Early_blight'

#List all files in the directory

file_list = os.listdir(directory_path)

# Print the list of files

print(file_list)

# Example: Get the first image file from the list

image_filename = file_list[0]

#Construct the full path

image_path = os.path.join(directory_path, image_filename)

# Print the full path to the image

print(image_path)

import tensorflow as tf
import numpy as np

#Function to preprocess the image

def preprocess_image(image_path, target_size=(150, 150)):
  img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, axis=0) # Expand dimensions to create batch of 1
  img_array = img_array / 255.0 # Rescale pixel values
  return img_array

# Load the model on CPU to avoid memory issues

with tf.device('/CPU:0'):
  model = tf.keras.models.load_model('potato_disease_classifier_aco.h5')

#Path to your image

image_path = '/content/drive/MyDrive/potato_dataset/train/Potato_Early_blight/d825093a-2bd7-458d-a9a1-036db6c08dec_RS_Ea'

# Preprocess the image

preprocessed_img = preprocess_image(image_path)

# Make prediction

with tf.device('/CPU:0'):
  predictions = model.predict(preprocessed_img)

#Get the class with the highest probability

predicted_class_index = np.argmax(predictions)

# Define the class labels (assuming these are the classes in your dataset)

class_labels = ['Potato_Early_blight', 'Potato_healthy', 'Potato_Late_blight']

#Get the predicted label

predicted_label = class_labels[predicted_class_index]

#Print the predicted label

print(f'Predicted label: {predicted_label}')

# If you want to visualize the image along with the label, you can use matplotlib

import matplotlib.pyplot as plt

# Load the image for visualization

img = tf.keras.preprocessing.image.load_img(image_path)

# Display the image with the predicted label

plt.imshow(img)
plt.title(f'Predicted: {predicted_label}')
plt.axis('off')
plt.show()

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model

with tf.device('/cpu:0'):
  model_path = 'potato_disease_classifier_aco.h5'
  model = tf.keras.models.load_model(model_path)

#Paths to the validation directories

val_dir = '/content/drive/MyDrive/potato_dataset/validation'

# Define the image size and batch size

img_height = 150
img_width = 150
batch_size = 32

# Create a data generator for the validation set

val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False # Important to keep the order of images and labels
)

# Ensure that predictions are also done on the CPU

with tf.device('/cpu:0'):
  #Predict on the validation set
  predictions = model.predict(validation_generator)

  #Get the predicted class indices
  predicted_class_indices = np.argmax(predictions, axis=1)

  #Get the true class indices
  true_class_indices = validation_generator.classes

  #Define the class labels
  class_labels = list(validation_generator.class_indices.keys())

  # Generate the classification report
  report = classification_report(true_class_indices, predicted_class_indices, target_names=class_labels)
  print(report)

  # Generate the confusion matrix
  conf_matrix = confusion_matrix(true_class_indices, predicted_class_indices)

  #Plot the confusion matrix
  plt.figure(figsize=(5, 4))
  sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Confusion Matrix')
  plt.show()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
import logging
from datetime import datetime
import time
import random

# Define the DenseNet-based model
def create_model(params):
  base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(params['dense_units'], activation='relu')(x)
  x = Dropout(params['dropout'])(x)
  predictions = Dense(3, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)

  # Freeze the base_model layers during initial training
  for layer in base_model.layers:
    layer.trainable = False

  model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

class Ant:
  def __init__(self):
    self.position = None
    self.cost = float('inf')

def random_params():
  return {
      'dense_units': np.random.choice([128, 256, 512]),
      'dropout': np.random.uniform(0.2, 0.5),
      'learning_rate': np.random.choice([1e-3, 1e-4, 1e-5])
  }

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

#ACO optimization function with detailed progress tracking
def aco_optimize(num_ants, num_generations):
  best_ant = Ant()
  start_time = time.time()

  for generation in range(num_generations):
    logging.info(f'Starting generation {generation + 1}/{num_generations}')
    ants = [Ant() for _ in range(num_ants)]

    for i, ant in enumerate(ants):
      ant.position = random_params()
      model = create_model(ant.position)
      logging.info(f'Generation {generation + 1}/{num_generations}, Ant {i + 1}/{num_ants}, Starting training')
      start_training_time = time.time()
      history = model.fit(train_generator, epochs=5, validation_data=validation_generator, verbose=1)
      end_training_time = time.time()

      val_accuracy = history.history['val_accuracy'][-1]
      ant.cost = -val_accuracy

      if ant.cost < best_ant.cost:
        best_ant.position = ant.position
        best_ant.cost = ant.cost

      #Print progress for each ant
      logging.info(f'Generation {generation + 1}/{num_generations}, Ant {i + 1}/{num_ants}, Best Cost: {-best_ant.cost}')
      logging.info(f'Training time for this ant: {(end_training_time - start_training_time)/60:.2f} minutes')

    logging.info(f'Completed generation {generation + 1}/{num_generations}, Best Cost: {-best_ant.cost:.4f}')

    #Estimate remaining time
    elapsed_time = time.time() - start_time
    remaining_generations = num_generations - (generation + 1)
    estimated_time_remaining = (elapsed_time / (generation + 1)) * remaining_generations
    logging.info(f'Elapsed Time: {elapsed_time/60:.2f} minutes, Estimated Time Remaining: {estimated_time_remaining/60:.2f} minutes')

  return best_ant.position

# Image dimensions
img_height = 150
img_width = 150
batch_size = 32


