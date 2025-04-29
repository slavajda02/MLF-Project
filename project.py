from utils.projectUtils import load_dataset, oversample_dataset

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

#SKlearn importsl
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler

#Kaggle import
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

#TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large



# Load the dataset
train, test, train_labels, test_labels = load_dataset('Dataset/')
# Split the dataset into training and validation sets to prevent overfitting
train, val, train_labels, val_labels = train_test_split(
    train, train_labels, test_size=0.1, stratify=train_labels['target']
)
train_labels = train_labels.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

print('\n')
print('Before oversampling and splitting:')
print('Train input', train.shape)
print('Train labels:', train_labels.shape)
print('Validation input:', val.shape)
print('Validation labels:', val_labels.shape)

# Oversample training data
train, train_labels = oversample_dataset(train, train_labels, noise=False)

# Remove the ID column
val_labels = val_labels.drop(columns=['ID'])

# Convert labels to int
train_labels['target'] = train_labels['target'].astype(int)

# Shuffle the training data
permutation = np.random.permutation(len(train))
train = train[permutation]
train_labels = train_labels.iloc[permutation].reset_index(drop=True)

# Convert labels to one-hot encoding
all_classes = sorted(set(train_labels['target']).union(set(test_labels['target'])))
train_labels_coded = pd.get_dummies(train_labels['target']).reindex(columns=all_classes, fill_value=False)
val_labels_coded = pd.get_dummies(val_labels['target']).reindex(columns=all_classes, fill_value=False)

# Print shapes of the datasets
print('\n')
print('After oversampling and splitting:')
print('Train input', train.shape)
print('Train labels:', train_labels_coded.shape)
print('Validation input:', val.shape)
print('Validation labels:', val_labels_coded.shape)
print('Testing input:', test.shape)


# Creating a model
#Convolutional Neural Network
model = Sequential([
    InputLayer(input_shape=(72, 48, 1)),
    Conv2D(16, (4, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(32, (4, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(3, activation='softmax'),
])

batch_size = 8
n_of_epochs = 100
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
optim = Adam(learning_rate=0.001)

model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


#Train the model
history = model.fit(train, train_labels_coded,
                     validation_data=(val, val_labels_coded), 
                     batch_size=batch_size, 
                     epochs=n_of_epochs,
                     callbacks=[lr_reducer, early_stopping],)

# Plot the training loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot the training accuracy
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()



# Predict on the test set
results = model.predict(test)
# Format results into same format as labels with one collumn
results = pd.DataFrame(results, columns=all_classes)
results['target'] = results.idxmax(axis=1)
results = results[['target']]
results = results.astype(int)
results = pd.concat([test_labels['ID'], results], axis=1)

# Save the results to a CSV file
results.to_csv('submission.csv', index=False)