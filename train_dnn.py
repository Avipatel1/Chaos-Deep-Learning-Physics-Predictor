import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists("chaos_training_data.txt"):
    print("Error: chaos_training_data.txt missing. Execute C++ generator first.")
    exit()

# 1. Load and Parse the Telemetry Dataset
raw_data = np.loadtxt("chaos_training_data.txt", dtype=np.float32)

# Input (X): State at time step 't' -> [theta1, theta2, omega1, omega2]
# Target (Y): State at time step 't+1'
X = raw_data[:-1]
Y = raw_data[1:]

# Split into 80% Training Data and 20% Evaluation Test Data
split_idx = int(len(X) * 0.8)
X_train, X_test = torch.tensor(X[:split_idx]), torch.tensor(X[split_idx:])
Y_train, Y_test = torch.tensor(Y[:split_idx]), torch.tensor(Y[split_idx:])

# 2. Build the Naive Deep Neural Network Architecture
class NaiveDNN(nn.Module):
    def __init__(self):
        super(NaiveDNN, self).__init__()
        # 4 inputs -> 3 hidden layers of 64 nodes -> 4 outputs
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
    def forward(self, x):
        return self.network(x)

model = NaiveDNN()
criterion = nn.MSELoss() # Standard Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. The Model Training Loop
print("Training the Naive DNN over 100 epochs...")
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, Y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100] | Training Loss: {loss.item():.6f}")

# 4. Testing the System: The Chaos Breakdown Test
model.eval()
print("\nEvaluating model performance on unseen test data...")

# Let's pick a starting point from our test set and see how it performs 
# when it has to predict sequentially into the future (Autoregressive loop)
current_state = X_test[0].clone()
true_trajectory = Y_test[:300].numpy()
predicted_trajectory = []

with torch.no_grad():
    for _ in range(300):
        # Predict the next step based on its OWN previous prediction
        next_state = model(current_state)
        predicted_trajectory.append(next_state.numpy())
        current_state = next_state # Feed prediction back in as input

predicted_trajectory = np.array(predicted_trajectory)

# 5. Plotting the Tracking Failure
plt.figure(figsize=(10, 5), facecolor='#0B0C10')
ax = plt.axes()
ax.set_facecolor('#0B0C10')
ax.tick_params(colors='white')

# Plot the true angle of Pendulum 2 vs the AI's naive guesses
plt.plot(true_trajectory[:, 1], label="Actual Physics (C++)", color='#2B82C9', linewidth=2)
plt.plot(predicted_trajectory[:, 1], label="Naive DNN Prediction", color='#E05A47', linestyle='--', linewidth=2)

plt.title("The Chaos Drift: Naive DNN Prediction Failure Over Time", color='white', pad=15)
plt.xlabel("Time Steps", color='white')
plt.ylabel("Pendulum 2 Angle (Radians)", color='white')
plt.legend(facecolor='#1F2833', labelcolor='white')
plt.grid(True, color='#1F2833', linestyle='--', alpha=0.5)
plt.show()
