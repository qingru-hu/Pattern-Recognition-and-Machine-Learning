import numpy as np


# Define the ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Define the derivative of the ReLU function
def relu_derivative(x):
    return (x > 0).astype(float)

# Define the forward pass of the network
def forward(x, W0, W1, W2, b0, b1, b2):
    z1 = np.dot(x, W0) + b0
    a1 = relu(z1)
    z2 = np.dot(a1, W1) + b1
    a2 = relu(z2)
    z3 = np.dot(a2, W2) + b2
    y_hat = z3
    return y_hat, a1, a2

def compute_loss(x, y, W0, W1, W2, b0, b1, b2):
    y_hat, _, _ = forward(x, W0, W1, W2, b0, b1, b2)
    loss = np.mean((y_hat - y) ** 2) / 2
    return loss

# Define a function to compute gradients by definition
def compute_gradient_definition(x, y, W0, W1, W2, b0, b1, b2, delta=1e-5):
    grads = []
    for param in [W0, W1, W2, b0, b1, b2]:
        param_grad = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            original_val = param[idx]

            param[idx] = original_val + delta
            y_hat, _, _ = forward(x, W0, W1, W2, b0, b1, b2)
            loss_plus_delta = np.mean((y_hat - y) ** 2) / 2

            param[idx] = original_val - delta
            y_hat, _, _ = forward(x, W0, W1, W2, b0, b1, b2)
            loss_minus_delta = np.mean((y_hat - y) ** 2) / 2

            param_grad[idx] = (loss_plus_delta - loss_minus_delta) / (2 * delta)

            param[idx] = original_val
        # print(param_grad.shape)
        grads.append(param_grad)

    return grads

# Define the function for students to implement back-propagation
def compute_gradient(x, y, W0, W1, W2, b0, b1, b2, a1, a2):
    # print('x', x.shape)
    # print('y', y.shape)
    # print('w2', W2.shape)
    # print('b2', b2.shape)
    # print('a2', a2.shape)
    # print('w1', W1.shape)
    # print('b1', b1.shape)
    # print('a1', a1.shape)
    # print('w0', W0.shape)
    # print('b0', b0.shape)
    
    a0 = relu(x)
    
    z2 = np.dot(a2, W2) + b2
    pz2 = (z2 - y) / x.shape[0]
    pa2 = pz2.dot(W2.T)
    pW2 = a2.T.dot(pz2)
    pb2 = np.sum(pz2)

    z1 = np.dot(a1, W1) + b1
    pz1 = relu_derivative(z1)*pa2
    pa1 = pz1.dot(W1.T)
    pW1 = a1.T.dot(pz1)
    pb1 = np.sum(pz1)
    
    z0 = np.dot(a0, W0) + b0
    pz0 = relu_derivative(z0)*(pa1)
    pW0 = a0.T.dot(pz0)
    pb0 = np.sum(pz0)
    
    # print('pz2', pz2.shape)
    # print('pb2', pb2.shape)
    # print('pW2', pW2.shape)
    # print('pz1', pz1.shape)
    # print('pb1', pb1.shape)
    # print('pW1', pW1.shape)
    # print('pz0', pz0.shape)
    # print('pb0', pb0.shape)
    # print('pW0', pW0.shape)
    
    return [pW0,pW1,pW2,pb0,pb1,pb2]


# Dataset generation function
def generate_dataset(num_samples, input_dim):
    X = np.random.randn(num_samples, input_dim)
    y = np.linalg.norm(X, axis=1, ord=2).reshape(-1, 1)
    return X, y


# Gradient descent implementation for a norm prediction regression problem
def gradient_descent(x, y, learning_rate, epochs):
    np.random.seed(42)
    W0 = np.random.randn(x.shape[1], 50).astype(np.float32) / np.sqrt(x.shape[1])
    b0 = np.zeros(50)
    W1 = np.random.randn(50, 50).astype(np.float32) / np.sqrt(50)
    b1 = np.zeros(50)
    W2 = np.random.randn(50, 1).astype(np.float32) / np.sqrt(50)
    b2 = np.zeros(1)

    y_hat, a1, a2 = forward(x, W0, W1, W2, b0, b1, b2)
    grads_bp = compute_gradient(x, y, W0, W1, W2, b0, b1, b2, a1, a2)

    # Compute gradients using definition
    grads_def = compute_gradient_definition(x, y, W0, W1, W2, b0, b1, b2)

    # Print gradients
    parms = ['W0', 'W1', 'W2', 'b0', 'b1', 'b2']
    for _ in range(len(grads_def)):
        # print("%s Gradients computed using back-propagation: "%parms[_], grads_bp[_])
        # print("%s Gradients computed using definition: "%parms[_], grads_def[_])
        print(f"{parms[_]} diff {np.abs(grads_def[_] - grads_bp[_]).max()}")

    print("Please make sure all the difference are sufficiently small to go on")

    for epoch in range(epochs):
        # Forward pass
        y_hat, a1, a2 = forward(x, W0, W1, W2, b0, b1, b2)

        # Compute gradients using back-propagation
        grads_bp = compute_gradient(x, y, W0, W1, W2, b0, b1, b2, a1, a2)

        print(f"{epoch}: loss is {compute_loss(x, y, W0, W1, W2, b0, b1, b2)}")

        # Update parameters using gradients from back-propagation
        W0 -= learning_rate * grads_bp[0]
        W1 -= learning_rate * grads_bp[1]
        W2 -= learning_rate * grads_bp[2]
        b0 -= learning_rate * grads_bp[3]
        b1 -= learning_rate * grads_bp[4]
        b2 -= learning_rate * grads_bp[5]

    return W0, W1, W2, b0, b1, b2


# Generate dataset
X_train, y_train = generate_dataset(500, 10)

# Set learning rate and number of epochs
learning_rate = 1e-2
epochs = 100

# Train the network using gradient descent
W0, W1, W2, b0, b1, b2 = gradient_descent(X_train, y_train, learning_rate, epochs)

X_test, y_test = generate_dataset(100, 10)
test_loss = compute_loss(X_test, y_test, W0, W1, W2, b0, b1, b2)
print(f"Test loss is {test_loss}")