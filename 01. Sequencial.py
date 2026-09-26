# Goal: train the simplest possible neural network (a single neuron) to learn
# the relationship y = 2x - 1 from a handful of example points.

import torch  # Core PyTorch library: tensors, autograd (automatic gradients), etc.
import torch.nn as nn  # Neural-network building blocks: layers, loss functions
import torch.optim as optim  # Optimization algorithms (SGD, Adam, ...) that update model weights
import numpy as numpy  # NumPy (imported but not used in this script; usually written `import numpy as np`)

# Model
# nn.Sequential chains layers in order; here there is only one layer.
# nn.Linear(1, 1) = one input feature -> one output: computes y = w * x + b,
# where the weight w and bias b start as random values and are learned during training.
model = nn.Sequential(nn.Linear(1,1))

# Loss and optimizer
# MSELoss = Mean Squared Error: average of (prediction - target)^2; measures how wrong the model is.
criterion = nn.MSELoss()
# SGD = Stochastic Gradient Descent. It updates w and b using their gradients.
# model.parameters() hands it the tensors to update; lr (learning rate) = step size of each update.
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Data
# Inputs: 6 samples, each a 1-element row, so shape is (6, 1) = (batch_size, num_features).
xs = torch.tensor([[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
# Targets: follow y = 2x - 1 (e.g. x=3 -> y=5). The model must discover w≈2 and b≈-1.
ys = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]], dtype=torch.float32)

# Training
# Repeat 500 times (epochs); each epoch uses all 6 samples at once. `_` is the loop counter.
for _ in range(500):
    optimizer.zero_grad()  # Clear gradients from the previous step (PyTorch accumulates them by default)
    outputs = model(xs)  # Forward pass: compute predictions w * x + b for all inputs
    loss = criterion(outputs, ys)  # Compare predictions with true targets -> single loss value
    loss.backward()  # Backward pass: autograd computes d(loss)/dw and d(loss)/db
    optimizer.step()  # Update parameters: w -= lr * grad_w, b -= lr * grad_b
    print(f'Epoch {_ + 1}, Loss: {loss.item():.4f}')  # .item() turns the 1-element tensor into a Python float

# Predict
# torch.no_grad() turns off gradient tracking: faster and uses less memory; not needed for inference.
with torch.no_grad():
    # Ask the trained model for x = 10. The true answer is 2*10 - 1 = 19, so expect a value close to 19.
    predicted = model(torch.tensor([[10.0]], dtype=torch.float32))
    print(f'Prediction: {predicted.item():.4f}')  # Print the prediction to 4 decimal places
