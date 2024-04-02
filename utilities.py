
import matplotlib.pyplot as plt

def performance(history):

  """
  This function will plot the performance of the model 
  showing both Training and Validation loss and accuracy.

  """

  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(history.history['loss'], label='Training')
  plt.plot(history.history['val_loss'], label='Validation')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(loc = 'upper left')
  plt.title('Training and validation loss')

  plt.subplot(1, 2, 2)
  plt.plot(history.history['accuracy'], label='Training')
  plt.plot(history.history['val_accuracy'], label='Validation')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(loc = 'upper left')
  plt.title('Training and validation accuracy')

  plt.tight_layout()
  plt.show()



import tensorflow as tf
import numpy as np



def preprocess_dataset(dataset):
    
    """
    This function helps to preprocess the images and labels inside the training 
    and test set.

    """
    images_list = []
    labels_list = []

    for images, labels in dataset:
        # Convert RGB images to grayscale
        gr_images = tf.image.rgb_to_grayscale(images)
        # Append the grayscale images and labels to the lists
        images_list.append(gr_images.numpy())
        labels_list.append(labels.numpy())

    images_array = np.concatenate(images_list)
    labels_array = np.concatenate(labels_list)

    return np.array(images_array), np.array(labels_array)

def preprocess_train_dataset(train_ds):
    train_images, train_labels = preprocess_dataset(train_ds)
    return train_images, train_labels

def preprocess_test_dataset(test_ds):
    test_images, test_labels = preprocess_dataset(test_ds)
    return test_images, test_labels


def basic_model(remove_layer=False):
    
    """
    This function creates the base model for the training of the neural network.

    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(100, 100, 1)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten()
    ])
    if not remove_layer: # this will remove a layer from the base model to see if there is any improvement
        model.add(tf.keras.layers.Dense(128, activation='relu'))


    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


def drop_model(remove_layer=False, dropout_rate = 0.5):
    
    """
    This function adds a dropout layer to the model with the aim to reduce overfitting.

    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(100, 100, 1)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten()
    ])
    if not remove_layer:
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

  
def second_model(input_shape=(100, 100, 1), l2_strength=0.01):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer= tf.keras.regularizers.l2(l2_strength)),
        tf.keras.layers.Dropout(0.7),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


