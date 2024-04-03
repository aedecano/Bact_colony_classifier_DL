import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 18 * 18, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def main(args):
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(root=args.train_data_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = datasets.ImageFolder(root=args.val_data_path, transform=transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = CNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training the model
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_data)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), args.model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deep learning model for bacterial colony classification")
    parser.add_argument("--train_data_path", type=str, default="data/train", help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default="data/validation", help="Path to validation data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--model_save_path", type=str, default="bacteria_classifier.pth", help="Path to save trained model")
    args = parser.parse_args()

    main(args)
