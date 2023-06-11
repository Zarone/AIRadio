from neural_networks.components.base import BaseNetwork
import numpy as np
import neural_networks.components.config as config
from neural_networks.components.activations import relu, relu_derivative
from typing import Tuple, Callable

class VAE(BaseNetwork):
  
  def __init__(
      self, 
      encoder_layers: Tuple[int, ...], 
      decoder_layers: Tuple[int, ...], 
      activation=relu, 
      activation_derivative=relu_derivative, 
      activation_exceptions: dict[int, Callable]={}
  ) -> None:


    if (encoder_layers[-1] != decoder_layers[0]): 
      raise Exception("Initialized VAE with inconsistent latent space size")

    self.activation = activation
    self.activation_derivative = activation_derivative
    self.activation_exceptions = activation_exceptions
    self.init_coefficients(encoder_layers, decoder_layers)


  def init_coefficients(self, e_layers: Tuple[int, ...], d_layers: Tuple[int, ...]) -> None:
    self.encoder_layers = e_layers
    self.decoder_layers = d_layers
    self.latent_size = d_layers[0]

    min=-1
    max=1
    
    length = len(e_layers) + len(d_layers) - 2

    self.biases: np.ndarray = np.empty(length, dtype=np.ndarray)
    self.weights: np.ndarray = np.empty(length, dtype=np.ndarray)

    index = 0

    # Encoder layers
    for _ in range(0, len(e_layers)-2):
      self.biases[index] = config.rng.uniform(min, max, (e_layers[index+1], 1))
      self.weights[index] = config.rng.uniform(min, max, (e_layers[index+1], e_layers[index]))
      index+=1

    # Encoder to latent space
    self.biases[index] = config.rng.uniform(min, max, (e_layers[index+1]*2, 1))
    self.weights[index] = config.rng.uniform(min, max, (e_layers[index+1]*2, e_layers[index]))
    index+=1
    
    # Sample to Decoder
    self.biases[index] = config.rng.uniform(min, max, (d_layers[1], 1))
    self.weights[index] = config.rng.uniform(min, max, (d_layers[1], d_layers[0]))
    index+=1

    # Decoder layers
    for _ in range(0, len(d_layers)-2):
      self.biases[index] = config.rng.uniform(min, max, (d_layers[index+2-len(e_layers)], 1))
      self.weights[index] = config.rng.uniform(min, max, (d_layers[index+2-len(e_layers)], d_layers[index+1-len(e_layers)]))
      index+=1


  def vae_loss(self, y_true, y_pred, mu, log_var):
    print("y_true")
    print(y_true)
    print("y_pred")
    print(y_pred)
    print("mu")
    print(mu)
    print("log_var")
    print(log_var)

    print("np.square(y_true - y_pred)")
    print(np.square(y_true - y_pred))

    # Reconstruction loss
    reconstruction_loss = np.mean(np.square(y_true - y_pred)[0], axis=-1)

    print("reconstruction_loss")
    print(reconstruction_loss)

    print("1 + log_var - np.square(mu) - np.exp(log_var)")
    print(1 + log_var - np.square(mu) - np.exp(log_var))

    print("np.mean( (1 + log_var - np.square(mu) - np.exp(log_var)) )")
    print(np.mean( (1 + log_var - np.square(mu) - np.exp(log_var)) ))

    # Regularization term - KL divergence
    kl_loss = -0.5 * np.mean( (1 + log_var - np.square(mu) - np.exp(log_var)))

    print("kl_loss")
    print(kl_loss)

    # Total loss
    total_loss = reconstruction_loss + kl_loss

    return total_loss


  def encode(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    activations = np.empty(
      len(self.encoder_layers)-1,
      dtype=np.ndarray
    )

    z_values = np.empty(
      len(self.encoder_layers)-1,
      dtype=np.ndarray
    )

    i = 0

    for _ in range(0, len(activations)-1):
      last_activations = input if i==0 else activations[i-1]

      z_values[i], activations[i] = self.feedforward_layer(i, last_activations)
      i+=1
 
    last_activations = input if i==0 else activations[-2]

    # z_{i} = w * a_{i-1} + b
    z_values[i] = np.matmul(self.weights[i], last_activations) + self.biases[i]

    activations[i] = z_values[i]

    parameters_count = len(activations[i])//2

    return (activations, z_values, activations[-1][0:parameters_count], activations[-1][parameters_count:parameters_count*2])


  def decode(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    activations = np.empty(
      len(self.decoder_layers)-1,
      dtype=np.ndarray
    )

    z_values = np.empty(
      len(self.decoder_layers)-1,
      dtype=np.ndarray
    )

    i = 0

    for _ in range(0, len(activations)-1):
      last_activations = input if i==0 else activations[i-1]
      
      coef_index = i+len(self.encoder_layers)-1

      z_values[i], activations[i] = self.feedforward_layer(coef_index, last_activations)
      i+=1

    last_activations = input if i==0 else activations[-2]

    coef_index = i+len(self.encoder_layers)-1

    # z_{i} = w * a_{i-1} + b
    z_values[i] = np.matmul(self.weights[coef_index], last_activations) + self.biases[coef_index]

    activations[i] = z_values[i]

    return (z_values, activations)

  def gen(self, mu, log_variance) -> Tuple[np.ndarray, np.ndarray]:
      epsilon = np.random.randn(len(mu)).reshape(-1, 1)
      z = mu + np.exp(0.5 * log_variance) * epsilon
      return (z, epsilon)

  def train(self, _training_data: np.ndarray, max_epochs: int, batch_size:int=100, test_data: (np.ndarray|None)=None, learning_rate=0.05) -> None:
    training_data = np.array(_training_data, copy=True)
    per_epoch = len(training_data) // batch_size
    for i in range(max_epochs):
      np.random.shuffle(training_data)
      for j in range(per_epoch):
        index = j * batch_size
        batch = training_data[index:index+batch_size]
        loss = self.training_step(batch, learning_rate)
        print(f"Epoch {i}, Mini-Batch {j}: Loss = {loss}")
      if test_data:
        raise NotImplementedError("Testing not implemented")

  def training_step(self, batch, learning_rate):
    # This gradient is added to for each data point in the batch
    weight_gradient = np.empty(self.weights.shape, np.ndarray)
    loss = 0

    for i, _ in enumerate(weight_gradient):
      weight_gradient[i] = np.zeros(self.weights[i].shape)

    for data_point in batch:
      z1, a1, mu, log_variance = self.encode(data_point)
      generated, epsilon = self.gen(mu, log_variance)
      z2, a2 = self.decode(generated)

      # These are needed for some gradient calculations
      z_values = np.hstack((z1, z2))
      activations = np.hstack((a1, a2))
      activations[len(self.encoder_layers)-2] = generated

      # Partial Derivative of Loss with respect to the output activations
      dL_daL = (activations[-1] - data_point) * (2/len(activations[-1]))
      loss += self.vae_loss(data_point, activations[-1], mu, log_variance)

      # Loss Gradients with respect to z, for just the decoder
      decoder_gradients_z = np.empty((len(z_values)), np.ndarray)
      decoder_gradients_z[-1] = dL_daL * self.activation_derivative(z_values[-1])

      last_index = len(z_values) - 1
      first_index = len(self.encoder_layers) - 2

      # Backpropagate through Decoder
      for j in range(last_index, first_index, -1):
        if j != last_index:
          decoder_gradients_z[j] = np.matmul(
              self.weights[j+1].transpose(), decoder_gradients_z[j+1]
            ) * self.activation_derivative(z_values[j])
        weight_gradient[j] -= np.matmul(decoder_gradients_z[j], activations[j-1].transpose())


      # ∂L/∂mu = ∂L/∂z_n * ∂z_n/∂a_(n-1) * ∂a_(n-1)/∂mu + ∂L/∂D * ∂D/∂mu
      #        = ∂L/∂z_n * w_n           * 1            + 1     * mu/N
      #        = ∂L/∂z_n * w_n + mu/N
      dL_dmu = np.matmul(self.weights[first_index+1].transpose(), decoder_gradients_z[first_index+1]) + mu/self.latent_size

      # ∂L/∂sigma = ∂L/∂z_n * ∂z_n/∂a_(n-1) * ∂a_(n-1)/∂sigma + ∂L/∂D * ∂D/∂sigma
      #           = ∂L/∂z_n * w_n           * epsilon         + 1     * mu/N
      #           = ∂L/∂z_n * w_n * epsilon + mu/N
      dL_dsigma = np.matmul(self.weights[first_index+1].transpose(), decoder_gradients_z[first_index+1]) * epsilon + mu/self.latent_size

      last_index = first_index
      first_index = -1

      decoder_gradients_z[last_index] = np.vstack((dL_dmu, dL_dsigma))

      # Backpropagate through Encoder
      for j in range(last_index, first_index, -1):
        if j != last_index:
          decoder_gradients_z[j] = np.matmul(
              self.weights[j+1].transpose(), decoder_gradients_z[j+1]
            ) * self.activation_derivative(z_values[j])
        weight_gradient[j] -= np.matmul(decoder_gradients_z[j], activations[j-1].transpose())

    self.weights += learning_rate * weight_gradient
    return loss

