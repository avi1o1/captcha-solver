import csv
import torch
import torchvision.transforms as transforms
from file import CaptchaDataset, MemoryEfficientCaptchaGenerator

MODEL_PATH = './Task2/extreme.pth'
TEST_DIR = './Task2/TestSet'

def test_model():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MemoryEfficientCaptchaGenerator().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = CaptchaDataset(TEST_DIR, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    results = []
    char_correct = 0
    char_total = 0
    word_correct = 0
    word_total = 0
    
    with torch.no_grad():
        for images, labels, lengths in test_loader:
            images = images.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=2)
            
            for pred, true_text, length in zip(predicted, labels, lengths):
                pred_text = ''.join([test_dataset.index_to_char[idx.item()] 
                                   for idx in pred[:length]])
                true_text = ''.join([test_dataset.index_to_char[idx.item()] 
                                   for idx in true_text if idx.item() != 0])
                
                # Calculate metrics
                min_len = min(len(true_text), len(pred_text))
                char_correct += sum(1 for i in range(min_len) 
                                  if true_text[i] == pred_text[i])
                char_total += max(len(true_text), len(pred_text))
                
                if true_text == pred_text:
                    word_correct += 1
                word_total += 1
                
                results.append({
                    'actual': true_text,
                    'predicted': pred_text,
                    'correct': true_text == pred_text
                })
    
    # Calculate accuracies
    char_accuracy = (char_correct / char_total) * 100
    word_accuracy = (word_correct / word_total) * 100
    
    # Save results
    with open('./Task2/test_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['actual', 'predicted', 'correct'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Test Results:")
    print(f"Character Accuracy: {char_accuracy:.2f}%")
    print(f"Word Accuracy: {word_accuracy:.2f}%")
    
    return char_accuracy, word_accuracy

if __name__ == "__main__":
    test_model()