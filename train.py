import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ----- CONFIGURABLE HYPERPARAMETERS -----
batch_size    = 64      # how many samples per training batch
epochs        = 10      # number of full passes through the training set
learning_rate = 0.001   # step size for optimizer

# CNN architecture parameters
conv1_out    = 32       # number of filters in 1st conv layer
conv2_out    = 64       # number of filters in 2nd conv layer
kernel_size1 = 3        # spatial size of kernel in 1st conv
kernel_size2 = 3        # spatial size of kernel in 2nd conv
fc1_units    = 128      # number of neurons in the first fully-connected layer
dropout_p    = 0.25     # dropout probability

# ----------------------------------------

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layers
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size1, padding=1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size2, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_p)
        # fully connected layers
        self.fc1 = nn.Linear(conv2_out * 7 * 7, fc1_units)
        self.fc2 = nn.Linear(fc1_units, 10)  # 10 output classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, conv2_out * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))   # MNIST mean/std
    ])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} — loss: {avg_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Model saved to mnist_cnn.pth")

if __name__ == "__main__":
    main()
