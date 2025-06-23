import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

# Configuration
DATASET_PATH = "dataset.json"
MODELS_DIR = "saved_models"
BATCH_SIZE = 64
EPOCHS = 50  
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8 
INPUT_SIZE = 1000  
HIDDEN_SIZE = 128

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


class FingerprintClassifier(nn.Module):
    """Basic neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(FingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After two 2x pooling operations
        self.fc_input_size = conv_output_size * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
class ComplexFingerprintClassifier(nn.Module):
    """A more complex neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexFingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After three 2x pooling operations
        self.fc_input_size = conv_output_size * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size*2)
        self.bn4 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x



def train(model, train_loader, test_loader, criterion, optimizer, epochs, model_save_path):
    """Train a PyTorch model and evaluate on the test set.
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        model_save_path: Path to save the best model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(traces)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * traces.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Evaluation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for traces, labels in test_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * traces.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_accuracy = correct / total
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_accuracy)
        
        # Print status
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, '
              f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}')
        
        # Save the best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with accuracy: {best_accuracy:.4f}')
    
    return best_accuracy



def evaluate(model, test_loader, website_names):
    """Evaluate a PyTorch model on the test set and show classification report with website names.
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for testing data
        website_names: List of website names for classification report
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for traces, labels in test_loader:
            traces, labels = traces.to(device), labels.to(device)
            outputs = model(traces)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report with website names instead of indices
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=website_names,
        zero_division=1
    ))
    
    return all_preds, all_labels


class FingerprintDataset(Dataset):
    """Custom PyTorch Dataset for website fingerprinting traces."""
    
    def __init__(self, traces, labels):
        """
        Args:
            traces: List of trace data (each trace is a list of 1000 values)
            labels: List of corresponding labels (integers)
        """
        self.traces = torch.FloatTensor(traces)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        return self.traces[idx], self.labels[idx]


def load_dataset(dataset_path):
    """Load and preprocess the dataset from JSON file.
    
    Args:
        dataset_path: Path to the dataset JSON file
        
    Returns:
        traces: List of trace data
        labels: List of labels (integers)
        website_names: List of unique website names
        label_to_name: Dictionary mapping label indices to website names
    """
    print(f"Loading dataset from {dataset_path}...")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    traces = []
    labels = []
    websites = []
    
    # Check for inconsistent trace lengths
    trace_lengths = [len(entry['trace_data']) for entry in data]
    unique_lengths = set(trace_lengths)
    
    if len(unique_lengths) > 1:
        print(f"Warning: Found inconsistent trace lengths: {unique_lengths}")
        expected_length = max(unique_lengths)  # Use the most common length
        print(f"Filtering to keep only traces of length {expected_length}")
    else:
        expected_length = list(unique_lengths)[0]
    
    # Filter and load data
    original_count = len(data)
    for entry in data:
        if len(entry['trace_data']) == expected_length:
            traces.append(entry['trace_data'])
            labels.append(entry['website_index'])
            websites.append(entry['website'])
    
    print(f"Filtered {original_count} -> {len(traces)} traces (removed {original_count - len(traces)} incomplete traces)")
    
    # Get unique websites and create mapping
    unique_websites = sorted(list(set(websites)))
    website_to_label = {website: idx for idx, website in enumerate(unique_websites)}
    label_to_name = {idx: website for idx, website in enumerate(unique_websites)}
    
    # Convert website names to consistent labels (0, 1, 2, ...)
    consistent_labels = [website_to_label[website] for website in websites]
    
    print(f"Loaded {len(traces)} traces from {len(unique_websites)} websites:")
    for i, website in enumerate(unique_websites):
        count = consistent_labels.count(i)
        print(f"  {i}: {website} ({count} traces)")
    
    return traces, consistent_labels, unique_websites, label_to_name


def main():
    """Main function to train and evaluate the models."""
    print("Starting Website Fingerprinting Classification")
    print("=" * 50)
    
    # Load dataset
    traces, labels, website_names, label_to_name = load_dataset(DATASET_PATH)
    num_classes = len(website_names)
    
    print(f"\nDataset Info:")
    print(f"  Total traces: {len(traces)}")
    print(f"  Trace length: {len(traces[0])}")
    print(f"  Number of classes: {num_classes}")
    
    # Create dataset
    dataset = FingerprintDataset(traces, labels)
    
    # Split dataset into train and test
    print(f"\nSplitting dataset (train: {TRAIN_SPLIT}, test: {1-TRAIN_SPLIT})")
    
    # Use stratified split to ensure balanced representation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-TRAIN_SPLIT, random_state=42)
    train_indices, test_indices = next(sss.split(traces, labels))
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Testing samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Define models to train
    models_config = [
        {
            'name': 'Basic CNN',
            'model_class': FingerprintClassifier,
            'save_path': os.path.join(MODELS_DIR, 'basic_model.pth')
        },
        {
            'name': 'Complex CNN',
            'model_class': ComplexFingerprintClassifier,
            'save_path': os.path.join(MODELS_DIR, 'complex_model.pth')
        }
    ]
    
    results = {}
    
    # Train and evaluate each model
    for config in models_config:
        print(f"\n{'='*60}")
        print(f"Training {config['name']}")
        print(f"{'='*60}")
        
        # Initialize model
        model = config['model_class'](INPUT_SIZE, HIDDEN_SIZE, num_classes)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {config['name']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train the model
        print(f"\nTraining for {EPOCHS} epochs...")
        best_accuracy = train(
            model, train_loader, test_loader, 
            criterion, optimizer, EPOCHS, config['save_path']
        )
        
        # Load the best model and evaluate
        print(f"\nLoading best model and evaluating...")
        model.load_state_dict(torch.load(config['save_path']))
        predictions, true_labels = evaluate(model, test_loader, website_names)
        
        results[config['name']] = {
            'best_accuracy': best_accuracy,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    # Print comparison of results
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        print(f"{model_name}: Best Test Accuracy = {result['best_accuracy']:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['best_accuracy'])
    print(f"\nBest performing model: {best_model_name}")
    print(f"Best accuracy: {results[best_model_name]['best_accuracy']:.4f}")
    
    print(f"\nTraining completed! Models saved in '{MODELS_DIR}' directory.")
    print("You can use the saved models for inference on new website traces.")

if __name__ == "__main__":
    main()
