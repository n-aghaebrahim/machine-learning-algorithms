import numpy as np
import pandas as pd
import argparse
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm



def generate_random_data(num_samples=100):
    # Generate random features (X) and corresponding labels (y)
    X = np.random.rand(num_samples, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(num_samples)

    # Split the data into training and test sets (80% train, 20% test)
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test

# Argument parser
parser = argparse.ArgumentParser(description='Linear Regression')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--num_samples', type=int, default=100, help='Number of samples in the dataset')
parser.add_argument('--dataset', type=str, default=None, help='Path to the dataset file')
args = parser.parse_args()

if args.dataset:
    # Load dataset from file if provided
    dataset = pd.read_csv(args.dataset)  # Read dataset using Pandas DataFrame
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    num_test_samples = int(0.2 * len(dataset))  # Use 20% of dataset for test
    X_train, y_train = X[:-num_test_samples], y[:-num_test_samples]
    X_test, y_test = X[-num_test_samples:], y[-num_test_samples:]
else:
    # Generate random data
    X_train, y_train, X_test, y_test = generate_random_data(num_samples=args.num_samples)

# Create and fit the model
model = LinearRegression()
train_losses = []

for epoch in tqdm(range(args.epochs)):
    # Training the model for one epoch
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    print(predictions_train)

    # Calculate and log the mean squared error loss on training data
    train_loss = np.mean((y_train - predictions_train) ** 2)
    train_losses.append(train_loss)

    # Save the training progress plot
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss (Training)')
    plt.title('Training Progress - Linear Regression')
    plt.savefig('linear_regression_training_progress.png')
    plt.close()

print(train_losses)


# Save the final model's weights (optional)
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Predict using the trained model
new_data = np.array([[5, 6], [6, 7]])
new_data = X_test
predictions = model.predict(new_data)
print("Predictions:", predictions)
