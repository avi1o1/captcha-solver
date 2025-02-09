import os
import csv
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

# Memory-optimized hyperparameters
TRAIN_TEST_RATIO = 0.8
NUM_SAMPLES = 25000
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 32
MAX_LENGTH = 10
VOCAB_SIZE = 128
IMAGE_SIZE = 128
PLOT_COUNT = 11

# Paths to the datasets
dataset_dir = './Task2/TrainSet'

class CaptchaDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.images = []
        self.texts = []
        
        self.char_to_index = {chr(i): i for i in range(VOCAB_SIZE)}
        self.index_to_char = {i: chr(i) for i in range(VOCAB_SIZE)}
        
        # Load images from flat directory
        cnt = 0
        for img_name in os.listdir(dataset):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(dataset, img_name))
                # Get text label from filename without extension
                self.texts.append(os.path.splitext(img_name)[0])
                
                cnt += 1
                if cnt == NUM_SAMPLES:
                    break

    def text_to_tensor(self, text):
        # Convert text to indices, handle variable lengths
        indices = [self.char_to_index[c] for c in text[:MAX_LENGTH]]
        length = len(indices)
        # Pad with zeros
        padded = indices + [0] * (MAX_LENGTH - length)
        return torch.tensor(padded, dtype=torch.long), length

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.imread(img_name)
        if image is None:
            raise ValueError(f"Failed to load image: {img_name}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        
        if self.transform:
            image = self.transform(image)
        
        text = self.texts[idx]
        text_tensor, length = self.text_to_tensor(text)
        
        return image, text_tensor, length

    def __len__(self):
        return len(self.images)
    
class MemoryEfficientCaptchaGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Add batch normalization and dropout for stability
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
        )
        
        self.cnn_output_size = 256 * (IMAGE_SIZE // 16) * (IMAGE_SIZE // 16)
        self.rnn_hidden_size = 256
        
        # Add dropout to RNN
        self.rnn = nn.GRU(
            input_size=self.cnn_output_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # Add batch norm and dropout to final layer
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.rnn_hidden_size * 2),
            nn.Dropout(0.2),
            nn.Linear(self.rnn_hidden_size * 2, VOCAB_SIZE)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Add gradient clipping in forward pass
        x = self.cnn(x)
        x = torch.clamp(x, -100, 100)
        
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1).repeat(1, MAX_LENGTH, 1)
        
        rnn_out, _ = self.rnn(x)
        
        # Reshape for batch norm
        rnn_out = rnn_out.reshape(-1, self.rnn_hidden_size * 2)
        output = self.fc(rnn_out)
        output = output.reshape(batch_size, MAX_LENGTH, -1)
        
        return output

def train_model(model, train_loader, test_loader, criterion, optimizer):
    history = {'train_loss': [], 'val_loss': []}
    device = next(model.parameters()).device
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Add gradient scaler for mixed precision training
    device_type = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    scaler = torch.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for images, labels, lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Use mixed precision training
            with torch.amp.autocast(device_type=device_type):
                outputs = model(images)
                
                # Modified loss calculation with better stability
                loss = 0
                for i in range(MAX_LENGTH):
                    batch_loss = criterion(outputs[:, i, :], labels[:, i])
                    if not torch.isnan(batch_loss):  # Skip NaN losses
                        loss += batch_loss
                loss = loss / MAX_LENGTH
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected! Skipping batch.")
                continue
                
            # Use gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        # Rest of the training loop remains the same...
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels, lengths in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                # Same stable loss calculation for validation
                batch_val_loss = 0
                for i in range(MAX_LENGTH):
                    loss = criterion(outputs[:, i, :], labels[:, i])
                    if not torch.isnan(loss):
                        batch_val_loss += loss
                batch_val_loss = batch_val_loss / MAX_LENGTH
                
                val_loss += batch_val_loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f'Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), './Task2/extreme.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    return history

def evaluate_model(model, test_loader, dataset):
    model.eval()
    device = next(model.parameters()).device
    results = []
    
    with torch.no_grad():
        for images, texts, lengths in test_loader:
            images = images.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=2)
            
            for pred, true_text, length in zip(predicted, texts, lengths):
                pred_text = ''.join([dataset.index_to_char[idx.item()] 
                                   for idx in pred[:length]])
                true_text = ''.join([dataset.index_to_char[idx.item()] 
                                   for idx in true_text if idx.item() != 0])
                results.append({
                    'actual': true_text,
                    'predicted': pred_text,
                    'correct': pred_text == true_text
                })
            
            # Free up memory
            del outputs, predicted
            torch.cuda.empty_cache()
    
    # Calculate accuracies
    char_correct = 0
    char_total = 0
    word_correct = 0
    word_total = len(results)
    
    for result in results:
        actual = result['actual']
        pred = result['predicted']
        min_len = min(len(actual), len(pred))
        char_correct += sum(1 for i in range(min_len) if actual[i] == pred[i])
        char_total += max(len(actual), len(pred))
        if actual == pred:
            word_correct += 1
    
    char_accuracy = (char_correct / char_total) * 100
    word_accuracy = (word_correct / word_total) * 100
    
    # Save results
    results_dir = './Task2'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['actual', 'predicted', 'correct'])
        writer.writeheader()
        writer.writerows(results)
    
    return char_accuracy, word_accuracy

def main():
    # Set memory-efficient CUDA options
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    
    # Simple data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess data
    print("Loading dataset...")
    dataset = CaptchaDataset(dataset_dir, transform=transform)
    train_size = int(TRAIN_TEST_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model and training
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MemoryEfficientCaptchaGenerator().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Train and evaluate
    print("Starting training...")
    history = train_model(model, train_loader, test_loader, criterion, optimizer)
    print("Evaluating model...")
    char_accuracy, word_accuracy = evaluate_model(model, test_loader, dataset)
    
    # Plot results
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    plt.subplot(1, 2, 2)
    plt.bar(['Character', 'Word'], [char_accuracy, word_accuracy])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance')
    plt.suptitle(f"Character-level accuracy: {char_accuracy:.2f}% | Word-level accuracy: {word_accuracy:.2f}%")
    
    plt.tight_layout()
    #  ensure there is a directory ./Task2/Plots
    plt.savefig(f'./Task2/Plots/{PLOT_COUNT}.png')
    plt.show()
    
    print(f"Training completed.")
    print(f"Character-level accuracy: {char_accuracy:.2f}%")
    print(f"Word-level accuracy: {word_accuracy:.2f}%")

if __name__ == "__main__":
    main()