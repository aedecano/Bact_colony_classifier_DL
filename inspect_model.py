import torch
import argparse

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

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Inspect trained model parameters")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to the trained model")
    args = parser.parse_args()

    # Load the model
    model = CNN()
    model.load_state_dict(torch.load(args.model_path))

    # Print the model's state dictionary
    print(model.state_dict())
