import torch

# Load the model
model = CNN()  # Assuming CNN is your model class
model.load_state_dict(torch.load('model.pth'))

# Print the model's state dictionary
print(model.state_dict())
