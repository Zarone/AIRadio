import numpy as np

def vae_loss(y_true, y_pred, mu, log_var):
    # Reconstruction loss
    reconstruction_loss = np.sum(np.square(y_true - y_pred))

    # Regularization term - KL divergence
    kl_loss = -0.5 * np.sum(1 + log_var - np.square(mu) - np.exp(log_var))

    # Total loss
    total_loss = reconstruction_loss + kl_loss

    return total_loss


# Create synthetic test case with known values
y_true = np.array([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]])
# y_pred = np.array([[0.25, 0.45, 0.65], [0.35, 0.55, 0.75]])
y_pred = np.array([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]])

# Known values for approximate posterior distribution (encoded distribution)
mu = np.array([[0.0, 1.0], [0.0, 1.0]])
log_var = np.array([[1.5, 0.0], [0.0, 0.0]])

# Known values for prior distribution
prior_mu = np.array([[0.5, 0.5], [0.5, 0.5]])
prior_log_var = np.array([[0.1, 0.1], [0.1, 0.1]])

loss = vae_loss(y_true, y_pred, mu, log_var)

print("Loss:", loss)
