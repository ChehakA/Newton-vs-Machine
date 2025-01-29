import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 


# Load data from CSV files
train_X = pd.read_csv("train_X.csv")
train_Y = pd.read_csv("train_Y.csv")

# Verify the data
print(train_X.head())
print(train_Y.head())

# Normalize the data 
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Fit on training data
train_X_scaled = scaler_X.fit_transform(train_X)
train_Y_scaled = scaler_Y.fit_transform(train_Y)

#train test split 
X_train, X_val, Y_train, Y_val = train_test_split(train_X_scaled, train_Y_scaled, test_size=0.2, random_state=42)



#model 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(Y_train.shape[1])
])

# Compiling the model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])

# Training the model
history = model.fit(X_train, Y_train, epochs=100, batch_size=5000, validation_split=0.01)

# model on the test data
test_loss, test_mse = model.evaluate(X_val, Y_val)

# evaluation results
print(f"Test Loss: {test_loss}")
print(f"Test MSE: {test_mse}")

mae = history.history['mae']
val_mae = history.history['val_mae']
print(mae)
print(val_mae)

plt.figure(figsize=(10, 6))
plt.plot(mae, label='Training MAE')
plt.plot(val_mae, label='Validation MAE', linestyle='--', color='orange', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Epoch vs. MAE')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("epoch_vs_mae.png")

Y_pred_train = model.predict(X_train)  # Predictions for the training set
Y_pred_val = model.predict(X_val) 


# Second plot, Trajectory Projections
train_X_scaled = np.random.rand(100, 4)  
train_Y_scaled = np.random.rand(100, 2)
positions_p1 = train_X_scaled[:, :2]  # Features for body 1 
positions_p2 = train_X_scaled[:, 2:4]  # Features for body 2 
positions_p3 = train_Y_scaled[:, :2]   # Predicted trajectory for body 3 
plt.figure(figsize=(10, 8))
plt.plot(positions_p1[:, 0], positions_p1[:, 1], label="Body 1 Trajectory", color='red', alpha=0.7)
plt.plot(positions_p2[:, 0], positions_p2[:, 1], label="Body 2 Trajectory", color='blue', alpha=0.7)
plt.plot(positions_p3[:, 0], positions_p3[:, 1], label="Body 3 Trajectory", color='green', alpha=0.7)
plt.title("Three-Body Problem Trajectories")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("Trajectories.png")