import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch

from model import (
    load_model,
    Net,
    get_device
)

# Use best model path
BEST_MODEL_PATH = './checkpoints/best_model.pth'
LEGACY_PATH = './checkpoints/cifar_net.pth'


def main():
    """
    Main function to evaluate the trained model on test dataset.
    """

    batch_size = 128  # Match training batch size
    device = get_device()
    print(f"Using device: {device}")

    # Load test data
    trainloader, testloader, classes = load_model(batch_size=batch_size)

    # Initialize model
    net = Net().to(device)
    
    # Load best checkpoint if available, otherwise use legacy model
    try:
        checkpoint = torch.load(BEST_MODEL_PATH, weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from checkpoint (epoch {checkpoint['epoch']+1})")
        print(f"Best validation accuracy: {checkpoint['best_accuracy']:.2f}%")
    except FileNotFoundError:
        print(f"Best model not found, trying legacy model...")
        net.load_state_dict(torch.load(LEGACY_PATH, weights_only=True))
        print(f"Loaded legacy model from {LEGACY_PATH}")
    
    # Set model to evaluation mode
    net.eval()

    # Quick test on first batch
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    print('\nGroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(min(4, batch_size))))

    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

    print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(min(4, batch_size))))

    # Evaluate on full test set
    print("\n" + "="*60)
    print("Evaluating on full test set...")
    print("="*60)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    overall_accuracy = 100 * correct / total
    print(f'\nOverall Accuracy: {overall_accuracy:.2f}% ({correct}/{total})')

    # Per-class accuracy
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # Print per-class accuracy
    print("\n" + "="*60)
    print("Per-Class Accuracy:")
    print("="*60)
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'{classname:10s}: {accuracy:5.1f}% ({correct_count}/{total_pred[classname]})')
    print("="*60)


if __name__ == "__main__":
    main()
