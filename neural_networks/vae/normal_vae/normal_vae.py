from neural_networks.components.base import BaseNetwork
import numpy as np
import neural_networks.components.config as config
from neural_networks.components.activations import relu
from typing import Tuple, Callable

class VAE(BaseNetwork):
  
  def __init__(
      self, 
      encoder_layers: Tuple[int, ...], 
      decoder_layers: Tuple[int, ...], 
      activation=relu, 
      activation_exceptions: dict[int, Callable]={}
  ) -> None:


    if (encoder_layers[-1] != decoder_layers[0]): 
      raise Exception("Initialized VAE with inconsistent latent space size")

    self.activation = activation
    self.activation_exceptions = activation_exceptions
    self.init_coefficients(encoder_layers, decoder_layers)


  def init_coefficients(self, e_layers: Tuple[int, ...], d_layers: Tuple[int, ...]) -> None:
    self.encoder_layers = e_layers
    self.decoder_layers = d_layers

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
    # Reconstruction loss
    reconstruction_loss = np.mean(np.square(y_true - y_pred), axis=-1)

    # Regularization term - KL divergence
    kl_loss = -0.5 * np.mean(1 + log_var - np.square(mu) - np.exp(log_var), axis=-1)

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
      epsilon = np.random.randn(len(mu))
      z = mu + np.exp(0.5 * log_variance) * epsilon
      return (z, epsilon)

  def train(self, _training_data: np.ndarray, max_epochs: int, batch_size:int = 100, test_data: (np.ndarray|None)=None) -> None:
    training_data = np.array(_training_data, copy=True)
    per_epoch = len(training_data) // batch_size
    for _ in range(max_epochs):
      np.random.shuffle(training_data)
      for j in range(per_epoch):
        index = j * batch_size
        batch = training_data[index:index+batch_size]
        self.training_step(batch)
      if test_data:
        raise NotImplementedError("Testing not implemented")

  def training_step(self, batch):
    weight_gradient = np.empty(self.weights.shape, np.ndarray)
    for i, _ in enumerate(weight_gradient):
      weight_gradient[i] = np.zeros(self.weights[i].shape)

    for data_point in batch:
      z1, a1, mu, log_variance = self.encode(data_point)
      generated, epsilon = self.gen(mu, log_variance)
      z2, a2 = self.decode(generated)
      z_values = np.hstack((z1, z2))
      activations = np.hstack((a1, a2))

      print("Don't forget I put exit at the end")
      exit()

