import torch
import torch.nn as nn
from torchvision import datasets, transforms
from train import CNN  # assumes train.py is in same folder

batch_size = 1000

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    model.eval()

    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
