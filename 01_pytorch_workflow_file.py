# PyTorch Workflow: From Basics to Model Deployment

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1. PyTorch Basics
## 1.1 Tensor Creation and Operations
x = torch.tensor([1, 2, 3])
y = torch.arange(0, 5, 0.5)

## 1.2 Device Management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)

# 2. Data Preparation
## 2.1 Creating Synthetic Data
X = torch.linspace(-10, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + torch.randn_like(X) * 0.5

## 2.2 Train-Test Split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# 3. Model Creation
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression().to(device)

# 4. Training Loop
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def train(model, X, y, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')


train(model, X_train, y_train)


# 5. Model Evaluation
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        mse = criterion(predictions, y)
    return mse.item()


train_mse = evaluate(model, X_train, y_train)
test_mse = evaluate(model, X_test, y_test)
print(f'Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')


# 6. Visualization
def plot_results(X, y, model):
    plt.scatter(X.cpu().numpy(), y.cpu().numpy(), label='Data')
    X_plot = torch.linspace(-10, 10, 200).reshape(-1, 1).to(device)
    y_plot = model(X_plot)
    plt.plot(X_plot.cpu().numpy(), y_plot.cpu().detach().numpy(), 'r-', label='Model')
    plt.legend()
    plt.show()


plot_results(X, y, model)

# 7. Model Saving and Loading
torch.save(model.state_dict(), 'linear_model.pth')

new_model = LinearRegression()
new_model.load_state_dict(torch.load('linear_model.pth'))
new_model.eval()

# 8. Making Predictions
new_data = torch.tensor([[5.0], [-3.0], [2.5]]).to(device)
with torch.inference_mode():
    predictions = new_model(new_data)
print("Predictions:", predictions)

''' 
This method is less favorable, because it can make modifications on the existing model.
new_data = torch.tensor([[5.0], [-3.0], [2.5]]).to(device)
predictions = new_model(new_data)
print("Predictions:", predictions)
'''


# 9. Advanced Topics (Brief Introduction)
## 9.1 Using GPU (if available)
if torch.cuda.is_available():
    print("GPU is available!")

## 9.2 Data Loaders
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

## 9.3 Learning Rate Scheduling
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 10. Best Practices and Tips
# - Use torch.nn.Module for model creation
# - Utilize torch.utils.data for efficient data handling
# - Implement early stopping to prevent overfitting
# - Use model.eval() for inference
# - Employ torch.no_grad() when computing validation loss