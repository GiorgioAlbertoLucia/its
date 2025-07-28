'''
    Common classes and functions for different neural network architectures.
'''

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NNRoutine:
    def __init__(self, model):

        self.model = model
        self.trainig_loss = []

    def run_training_loop(self, train_loader: DataLoader, learning_rate:float=1e-3, number_epochs:int=10):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(number_epochs):
            print(f"Epoch {epoch + 1}/{number_epochs}")
            self.model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"\tLoss: {epoch_loss:.4f}")
            self.training_losses.append(epoch_loss)

        return self.training_losses

    def plot_loss(self, outfile:str):
        """
        Plots the training loss over epochs.

        Parameters:
        losses (list): List of loss values for each epoch.
        """

        plt.plot(self.training_losses)
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(outfile)

    def run_model_eval(self, test_loader: DataLoader):
        """
        Evaluates the model on the test set.

        Parameters:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.

        Returns:
        accuracy (float): The accuracy of the model on the test set.
        """

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

class ParticleDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def torch_data_preparation(X, y, test_val_size:float=0.2, batch_size:int=128):
    """
    Prepares the data for training by separating features and target labels.

    Parameters:
    X (np.ndarray): Feature data.
    y (np.ndarray): Target labels.

    Returns:
    train_loader (DataLoader): DataLoader for the training set.
    test_loader (DataLoader): DataLoader for the test set.
    label_encoder (LabelEncoder): LabelEncoder fitted to the target labels.
    """

    print('Preparing data for training.')

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_val_size, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

    train_dataset = ParticleDataset(X_train, y_train)
    val_dataset = ParticleDataset(X_val, y_val)
    test_dataset = ParticleDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def downsample_class(df: pd.DataFrame, target_class: str):

    print(f"Downsampling class {target_class} to match the size of the other classes.")

    df_minority = df[df['fPartID'] != target_class]
    df_majority = df[df['fPartID'] == target_class].sample(
        n=len(df_minority), random_state=42
    )
    return pd.concat([df_minority, df_majority])

