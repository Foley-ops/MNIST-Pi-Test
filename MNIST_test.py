import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import psutil  # For CPU power and memory usage

# Device setup (Pi likely runs on CPU)
device = torch.device("cpu")

# Data Preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # Smaller batch size for Pi
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# Simple Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Performance: Training Time and CPU Usage
start_time = time.time()

# Training Loop
epochs = 3  # Reduce epochs for faster testing on Pi
for epoch in range(epochs):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

training_time = time.time() - start_time
print(f"Training Time: {training_time:.2f} seconds")

# CPU Usage Measurement Start
psutil.cpu_percent(interval=None)  # Reset the CPU measurement

# Inference Performance Measurement
model.eval()
inference_times = []
correct = 0
total = 0

start_inference_total = time.time()

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # Measure inference time per batch
        start_inference = time.time()
        output = model(data)
        end_inference = time.time()

        inference_times.append(end_inference - start_inference)

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

# CPU Usage Measurement End
cpu_usage = psutil.cpu_percent(interval=None)  # Capture CPU usage during inference

end_inference_total = time.time()
total_inference_time = end_inference_total - start_inference_total

# Average Inference Time
avg_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time per Batch: {avg_inference_time:.6f} seconds")
print(f"Total Inference Time: {total_inference_time:.2f} seconds")

# Accuracy
accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")

# CPU and Memory Usage
print(f"CPU Usage during Inference: {cpu_usage}%")
mem_usage = psutil.virtual_memory().percent
print(f"Memory Usage: {mem_usage}%")
