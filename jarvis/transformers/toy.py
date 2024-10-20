import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CarTransformer(nn.Module):
    def __init__(self, num_attributes=3, d_model=64, num_heads=4, num_layers=2):
        super(CarTransformer, self).__init__()

        # Token embedding (for x, y, and speed of each car)
        self.token_embedding = nn.Linear(num_attributes, d_model)

        # Positional embedding (in case you want to encode positional info)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 100, d_model))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # Output layers for steering, throttle, and brake
        self.fc_steering = nn.Linear(d_model, 1)
        self.fc_throttle = nn.Linear(d_model, 1)
        self.fc_brake = nn.Linear(d_model, 1)

    def forward(self, cars_attributes):
        # cars_attributes: [batch_size, num_cars, num_attributes]

        # Step 1: Embed the car attributes into a higher-dimensional space
        x = self.token_embedding(cars_attributes)

        # Step 2: Add positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]

        # Step 3: Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Step 4: Aggregate the output (simple mean here, but you can apply other techniques)
        x = torch.mean(x, dim=1)  # [batch_size, d_model]

        # Step 5: Predict control actions
        steering = self.fc_steering(x).squeeze(-1)  # [batch_size]
        throttle = self.fc_throttle(x).squeeze(-1)  # [batch_size]
        brake = self.fc_brake(x).squeeze(-1)  # [batch_size]

        return steering, throttle, brake


# Create a toy batch of data: 5 cars, 3 attributes (x, y, speed), batch size of 2
batch_size = 2
num_cars = 5
num_attributes = 3
# [batch_size, num_cars, num_attributes]
cars_attributes = torch.randn(batch_size, num_cars, num_attributes)

# Instantiate the model and pass the input through it
model = CarTransformer(num_attributes=num_attributes)
steering, throttle, brake = model(cars_attributes)

print("Steering:", steering)
print("Throttle:", throttle)
print("Brake:", brake)


# Assuming you have ground truth data
steering_gt = torch.randn(batch_size)
throttle_gt = torch.randn(batch_size)
brake_gt = torch.randn(batch_size)

# Mean Squared Error loss for each control output
loss_fn = nn.MSELoss()

loss_steering = loss_fn(steering, steering_gt)
loss_throttle = loss_fn(throttle, throttle_gt)
loss_brake = loss_fn(brake, brake_gt)

# Total loss
total_loss = loss_steering + loss_throttle + loss_brake

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    steering, throttle, brake = model(cars_attributes)

    # Calculate the loss
    loss_steering = loss_fn(steering, steering_gt)
    loss_throttle = loss_fn(throttle, throttle_gt)
    loss_brake = loss_fn(brake, brake_gt)
    total_loss = loss_steering + loss_throttle + loss_brake

    # Backpropagation
    total_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {total_loss.item()}")


# Test data (similar to training data, but unseen during training)
# test_cars_attributes = torch.randn(
#     batch_size, num_cars, num_attributes)  # Random test data
# steering_gt_test = torch.randn(batch_size)
# throttle_gt_test = torch.randn(batch_size)
# brake_gt_test = torch.randn(batch_size)

test_cars_attributes = cars_attributes
steering_gt_test = steering_gt
throttle_gt_test = throttle_gt
brake_gt_test = brake_gt


model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    # Forward pass for test data
    steering_pred, throttle_pred, brake_pred = model(test_cars_attributes)

# Mean Squared Error loss for each control output
loss_fn = nn.MSELoss()

loss_steering_test = loss_fn(steering_pred, steering_gt_test)
loss_throttle_test = loss_fn(throttle_pred, throttle_gt_test)
loss_brake_test = loss_fn(brake_pred, brake_gt_test)

# Total test loss
total_test_loss = loss_steering_test + loss_throttle_test + loss_brake_test
print(f"Test Loss: {total_test_loss.item()}")


def mae(pred, gt):
    return torch.mean(torch.abs(pred - gt))


steering_mae = mae(steering_pred, steering_gt_test)
throttle_mae = mae(throttle_pred, throttle_gt_test)
brake_mae = mae(brake_pred, brake_gt_test)

print(f"MAE Steering: {steering_mae.item()}")
print(f"MAE Throttle: {throttle_mae.item()}")
print(f"MAE Brake: {brake_mae.item()}")


# Plot steering predictions vs. ground truth
plt.plot(steering_gt_test.numpy(), label='Ground Truth Steering')
plt.plot(steering_pred.numpy(), label='Predicted Steering')
plt.legend()
plt.title("Steering Predictions vs Ground Truth")
plt.show()
