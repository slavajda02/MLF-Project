# Heper functions for loading and processing datasets for MLF final project
# @Miloslav Kužela (240648)

import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

def load_dataset(dir_path):
    """
    Load dataset from a given directory path. Normalizes and reshapes the data for the machine learning model.
    
    Args:
        dir_path (str): Path to the dataset directory.
    
    Returns:
        tuple: A tuple containing:
            - training_dataset (np.array): Array of training data.
            - testing_dataset (np.array): Array of testing data.
            - label_train (pd.DataFrame): DataFrame of training labels.
            - label_test (pd.DataFrame): DataFrame of testing labels.
    """
    
    train_path = os.path.join(dir_path, 'Train')
    test_path = os.path.join(dir_path, 'Test')
    
    label_training = os.path.join(dir_path, 'label_train.csv')
    label_testing = os.path.join(dir_path, 'test_format.csv')
    
    # List directories and sort them by their ID
    training_files = sorted(os.listdir(train_path), key = lambda x: int(x.split('.')[0]))
    testing_files = sorted(os.listdir(test_path), key = lambda x: int(x.split('.')[0]))
    
    # Load training and testing datasets
    training_dataset = []
    for path in training_files:
        if path.endswith('.npy'):
            train_data = np.load(os.path.join(train_path, path))
            training_dataset.append(train_data)
            
    testing_dataset = []
    for path in testing_files:
        if path.endswith('.npy'):
            test_data = np.load(os.path.join(test_path, path))
            testing_dataset.append(test_data)
    
    # Load labels
    label_train = pd.read_csv(label_training)
    label_test = pd.read_csv(label_testing)
    
    # Converts lists to numpy arrays
    training_dataset = np.array(training_dataset)
    testing_dataset = np.array(testing_dataset)
    
    # Normalizes and reshapes the data
    train_normalized = (training_dataset - training_dataset.min()) / (training_dataset.max() - training_dataset.min())
    train_reshaped = train_normalized.reshape(-1, 72, 48, 1)
    
    testing_normalized = (testing_dataset - testing_dataset.min()) / (testing_dataset.max() - testing_dataset.min())
    testing_reshaped = testing_normalized.reshape(-1, 72, 48, 1)
    
    print(f"Training dataset shape: {training_dataset.shape}")
    print(f"Testing dataset shape: {testing_dataset.shape}")
    
    return train_reshaped, testing_reshaped, label_train, label_test

def oversample_dataset(dataset, labels, noise=False):
    """
    Oversample the dataset to balance the classes. With the abbility to add noise to the data.
    
    Args:
        dataset (np.array): The dataset to be oversampled.
        labels (pd.DataFrame): The labels corresponding to the dataset.
        noise (bool): If True, adds noise to the oversampled data.
    
    Returns:
        tuple: A tuple containing:
            - oversampled_dataset (np.array): The oversampled dataset.
            - oversampled_labels (pd.DataFrame): The oversampled labels.
    """
    
    # Count the number of samples in each class
    class_counts = labels['target'].value_counts()
    
    # Find the maximum class count
    max_count = class_counts.max()
    
    # Create a new list to hold the oversampled data
    oversampled_data = []
    oversampled_labels = []
    
    for label, count in class_counts.items():
        # Get the indices of the samples for this label
        indices = labels[labels['target'] == label].index.tolist()
        
        # Randomly sample with replacement to reach max_count
        sampled_indices = np.random.choice(indices, max_count, replace=True)
        
        # Append the sampled data and labels to the new lists
        for idx in sampled_indices:
            if noise:
                # Add noise to the data
                noise_data = dataset[idx] + np.random.normal(0, 0.01, dataset[idx].shape)
                oversampled_data.append(noise_data)
                oversampled_labels.append(labels.iloc[idx])
            else:
                oversampled_data.append(dataset[idx])
                oversampled_labels.append(labels.iloc[idx])
    
    return np.array(oversampled_data), pd.DataFrame(oversampled_labels)

def oversample_synthetic(train_data, train_labels):
    """
    Apply SMOTE to the training data to balance the classes.
    
    Args:
        train_data (np.array): The training data.
        train_labels (pd.DataFrame): The training labels.
    
    Returns:
        tuple: A tuple containing:
            - X_resampled (np.array): The oversampled training data.
            - y_resampled (pd.DataFrame): The oversampled training labels.
    """
    
   # Oversampling the training dataset using SMOTE
    SMOTE = SMOTE(random_state=42)
    # Flatten training data
    train_flat = train.reshape(train.shape[0], -1)
    # Apply SMOTE
    train_flat, train_labels = SMOTE.fit_resample(train_flat, train_labels['target'])
    # Unflatten
    train = train_flat.reshape(-1, 72, 48, 1)
    # Convert labels to DataFrame
    train_labels = pd.DataFrame(train_labels, columns=['target'])
    
    return train, train_labels