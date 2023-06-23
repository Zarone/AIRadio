from neural_networks.components.optimizer.adam import Adam
from neural_networks.components.base import BaseNetwork
import numpy as np
import numpy.typing as npt
import neural_networks.components.config as config
from neural_networks.components.activations import *
from typing import Tuple, Callable, Any, List
import matplotlib.pyplot as plt
import math
from neural_networks.components.optimizer.optimizer import Optimizer
from neural_networks.components.optimizer.SGD import SGD

class VAE(BaseNetwork):
  
  def __init__(
      self, 
      encoder_layers: Tuple[int, ...], 
      decoder_layers: Tuple[int, ...], 
      activation=leaky_relu, 
      activation_derivative=leaky_relu_derivative, 
      activation_exceptions: dict[int, Callable]={},
      optimizer: Optimizer = Adam()
  ) -> None:


    if (encoder_layers[-1] != decoder_layers[0]): 
      raise Exception("Initialized VAE with inconsistent latent space size")

    self.optimizer = optimizer
    self.activation = activation
    self.activation_derivative = activation_derivative
    self.activation_exceptions = activation_exceptions
    self.init_coefficients(encoder_layers, decoder_layers)

    self.weight_gradient = None
    self.bias_gradient = None


  def init_coefficients(self, e_layers: Tuple[int, ...], d_layers: Tuple[int, ...]) -> None:
    self.encoder_layers = e_layers
    self.decoder_layers = d_layers
    self.latent_size = d_layers[0]

    length = len(e_layers) + len(d_layers) - 2

    self.biases: np.ndarray = np.empty(length, dtype=np.ndarray)
    self.weights: np.ndarray = np.empty(length, dtype=np.ndarray)

    index = 0

    # Encoder layers
    for _ in range(0, len(e_layers)-2):
      max = math.sqrt(2 / e_layers[index])
      min = -max
      self.biases[index] = config.rng.uniform(min, max, (e_layers[index+1], 1))
      self.weights[index] = config.rng.uniform(min, max, (e_layers[index+1], e_layers[index]))
      index+=1

    max = math.sqrt(2 / e_layers[index])
    min = -max

    # Encoder to latent space
    self.biases[index] = config.rng.uniform(min, max, (e_layers[index+1]*2, 1))
    self.weights[index] = config.rng.uniform(min, max, (e_layers[index+1]*2, e_layers[index]))
    index+=1
    
    max = math.sqrt(2 / d_layers[0])
    min = -max

    # Sample to Decoder
    self.biases[index] = config.rng.uniform(min, max, (d_layers[1], 1))
    self.weights[index] = config.rng.uniform(min, max, (d_layers[1], d_layers[0]))
    index+=1

    # Decoder layers
    for _ in range(0, len(d_layers)-2):
      max = math.sqrt(2 / d_layers[index+1-len(e_layers)])
      min = -max
      self.biases[index] = config.rng.uniform(min, max, (d_layers[index+2-len(e_layers)], 1))
      self.weights[index] = config.rng.uniform(min, max, (d_layers[index+2-len(e_layers)], d_layers[index+1-len(e_layers)]))
      index+=1


  def loss(self, y_true, y_pred, mu=np.array([0.0]), log_var=np.array([0.0])):
    n = y_true.shape[0]  # Number of samples

    # Reconstruction loss
    reconstruction_loss = np.sum(np.square(y_true - y_pred)) / n

    # Regularization term - KL divergence
    kl_loss = -0.5 * np.sum(1 + log_var - np.square(mu) - np.exp(log_var)) / len(mu)

    return (reconstruction_loss, kl_loss)


  def encode(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray[float, Any], np.ndarray[float, Any]]:
    
    activations = np.array([None] * (len(self.encoder_layers) - 1), dtype=npt.NDArray[np.float256])
    z_values = np.array([None] * (len(self.encoder_layers) - 1))

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

    return (
      z_values,
      activations,
      activations[-1][:parameters_count], 
      activations[-1][parameters_count:parameters_count*2]
    )


  def decode(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    activations = np.array([None] * (len(self.decoder_layers) - 1))
    z_values = np.array([None] * (len(self.decoder_layers) - 1))

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

  def feedforward(self, input: np.ndarray):
    _, _, mu, log_variance = self.encode(input)
    generated, _ = self.gen(mu, log_variance)
    return self.decode(generated)[1][-1]

  def gen(self, mu, log_variance) -> Tuple[np.ndarray, np.ndarray]:
    epsilon = (np.random.randn(len(mu)).reshape(-1, 1))
    z = mu + np.exp(0.5 * log_variance) * epsilon
    return (z, epsilon)

  @staticmethod
  def graph_loss(losses, reconstruction_losses, kl_losses, test_losses = [], test_reconstruction_losses = None, test_kl_losses = None):
    sub = plt.subplots(2 if not len(test_losses) == 0 else 1, sharex=True)
    axs: Any = sub[1]
    if len(test_losses) == 0:
      axs = [axs]
    axs[0].plot(losses, "purple", label="Total Loss")
    axs[0].plot(kl_losses, "red", label="KL Divergence")
    axs[0].plot(reconstruction_losses, "blue", label="Reconstruction Loss")
    axs[0].set(ylabel='Loss',
     title='Loss over time (Training Data)')
    axs[0].legend(loc="upper left")
    axs[0].grid()
    axs[0].semilogy()

    if test_losses:
      axs[1].plot(test_losses, "purple", label="Total Loss")
      axs[1].plot(test_kl_losses, "red", label="KL Divergence")
      axs[1].plot(test_reconstruction_losses, "blue", label="Reconstruction Loss")
      axs[1].set(xlabel='Mini Batch', ylabel='Loss',
       title='Loss over time (Test Data)')
      axs[1].legend(loc="upper left")
      axs[1].grid()
      axs[1].semilogy() 

    plt.show()


  def train(self, _training_data: np.ndarray, max_epochs: int, batch_size:int=100, test_data: (np.ndarray|None)=None, learning_rate=0.05, graph=False, print_epochs=True) -> None:
    print("len sounds", len(_training_data))
    print("len test_data", len(test_data) if not test_data is None else 0)
    losses = []
    kl_losses = []
    reconstruction_losses = []
    test_losses = []
    test_kl_losses = []
    test_reconstruction_losses = []
    training_data = np.array(_training_data, copy=True)
    per_epoch = len(training_data) // batch_size

    if per_epoch == 0:
      raise Exception("Batch Size greater than Data Set")
    for i in range(max_epochs):
      np.random.shuffle(training_data)
      for j in range(per_epoch):
        index = j * batch_size
        batch = training_data[index:index+batch_size]

        reconstruction_loss, kl_loss = self.training_step(batch, learning_rate, print_epochs)

        kl_losses.append(kl_loss)
        reconstruction_losses.append(reconstruction_loss)
        loss = kl_loss+reconstruction_loss
        losses.append(loss)

        test_loss = 0
        if not(test_data is None) and print_epochs:
          test_reconstruction_loss = 0
          test_kl_loss = 0
          for index, element in enumerate(test_data):
            _, _, mu, logvar = self.encode(element)
            generated, _ = self.gen(mu, logvar)
            delta_test_reconstruction_loss, delta_test_kl_loss = self.loss(element, self.decode(generated)[1][-1], mu, logvar)
            test_reconstruction_loss += delta_test_reconstruction_loss
            test_kl_loss += delta_test_kl_loss
          test_loss = test_reconstruction_loss + test_kl_loss

          test_loss /= len(test_data)
          test_kl_loss /= len(test_data)
          test_reconstruction_loss /= len(test_data)

          test_losses.append(test_loss)
          test_kl_losses.append(test_kl_loss)
          test_reconstruction_losses.append(test_reconstruction_loss)

        if print_epochs:
          print(f"Epoch {i}, Mini-Batch {j}: Loss = {loss}, Test Loss = {test_loss}")
          # print(f"Epoch {i}, Mini-Batch {j}: KL Loss = {kl_loss}, Reconstruction Loss = {reconstruction_loss}")
    if graph:
      self.graph_loss(losses, reconstruction_losses, kl_losses, test_losses, test_reconstruction_losses, test_kl_losses)

  def training_step(self, batch, learning_rate, print_epochs):
    self.init_gradients()
    if (self.weight_gradient is None or self.bias_gradient is None):
      raise Exception("weight gradient not defined for some reason")
    reconstruction_loss = 0
    kl_loss = 0

    # Beta affects the relative importance of kl_loss 
    # with respect to reconstruction_loss in calculating
    # the gradient.
    Beta = 1


    for _, data_point in enumerate(batch):
      z1, a1, mu, log_variance = self.encode(data_point)
      generated, epsilon = self.gen(mu, log_variance)
      z2, a2 = self.decode(generated)

      # These are needed for some gradient calculations
      a1[len(self.encoder_layers)-2] = generated

      z_values = np.concatenate((z1, z2))
      activation_values = np.concatenate((a1, a2))

      # Partial Derivative of Loss with respect to the output activations
      dL_daL = (activation_values[-1] - data_point) * (2/len(activation_values[-1]))
      if print_epochs:
        delta_reconstruction_loss, delta_kl_loss = self.loss(data_point, activation_values[-1], mu, log_variance)
        reconstruction_loss += delta_reconstruction_loss
        kl_loss += delta_kl_loss

      len_z = len(z_values)

      # Loss Gradients with respect to z, for just the decoder
      decoder_gradients_z = np.array([None] * len_z)

      decoder_gradients_z[-1] = dL_daL

      last_index = len_z - 1
      first_index = len(self.encoder_layers) - 2

      # Backpropagate through Decoder
      for j in range(last_index, first_index, -1):
        z_layer = z_values[j]
        if j != last_index:
          decoder_gradients_z[j] = np.matmul(
              self.weights[j+1].transpose(), decoder_gradients_z[j+1]
            ) * self.activation_derivative(z_layer)
        a_layer = activation_values[j-1]
        self.weight_gradient[j] += np.matmul(decoder_gradients_z[j], a_layer.transpose())
        self.bias_gradient[j] += decoder_gradients_z[j]


      dL_da = np.matmul(self.weights[first_index+1].transpose(), decoder_gradients_z[first_index+1])

      # ∂L/∂mu = ∂L/∂z_n * ∂z_n/∂a_(n-1) * ∂a_(n-1)/∂mu + ∂L/∂D * ∂D/∂mu
      #        = ∂L/∂z_n * w_n           * 1            + 1    * mu/N
      #        = ∂L/∂z_n * w_n + mu/N
      dL_dmu = dL_da + Beta*mu/self.latent_size

      # ∂L/∂logvar = ∂L/∂z_n * ∂z_n/∂a_(n-1) * ∂a_(n-1)/∂logvar                + ∂L/∂D   * ∂D/∂logvar
      #            = ∂L/∂z_n * w_n           * epsilon/2*np.exp(logvar/2)      + 1       * (1-np.exp(logvar))/2N
      #            = ∂L/∂z_n * w_n * epsilon/2*np.exp(logvar/2) - (1-np.exp(logvar))/2N
      dL_dlogvar = \
        dL_da \
        * epsilon/2*np.exp(log_variance/2) \
        -Beta*(1-np.exp(log_variance))/self.latent_size/2

      last_index = first_index
      first_index = -1

      decoder_gradients_z[last_index] = np.concatenate((dL_dmu, dL_dlogvar), axis=0)

      # Backpropagate through Encoder
      for j in range(last_index, first_index, -1):
        if j != last_index:
          decoder_gradients_z[j] = np.matmul(
              self.weights[j+1].transpose(), decoder_gradients_z[j+1]
            ) * self.activation_derivative(z_values[j])

        if j != 0:
          self.weight_gradient[j] += np.matmul(decoder_gradients_z[j], activation_values[j-1].transpose())
          self.bias_gradient[j] += decoder_gradients_z[j]
      
      self.weight_gradient[0] += np.matmul(decoder_gradients_z[0], data_point.transpose())
      self.bias_gradient[0] += decoder_gradients_z[0]

    self.weights -= learning_rate/len(batch) * self.optimizer.adjusted_weight_gradient(self.weight_gradient)
    self.biases -= learning_rate/len(batch) * self.optimizer.adjusted_bias_gradient(self.bias_gradient)
    return reconstruction_loss/len(batch), kl_loss/len(batch)

