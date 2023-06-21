from neural_networks.components.optimizer.optimizer import Optimizer
from neural_networks.components.optimizer.adam import Adam
from typing import Any, Tuple, Callable, List
import numpy as np
import neural_networks.components.config as config
from neural_networks.components.activations import linear, linear_derivative, relu, relu_derivative
import math
import matplotlib.pyplot as plt

class BaseNetwork:

  """
  Parameters
  ----------
  layers 
    This defines the number of nodes in each activation layer (including input space and latent space)
  activation
    This is the primary activation function which the neural network uses
  activation_exceptions
    This is a dictionary where the key equals the layer where the exception occurs, and the key is the replacement activation function
  """
  def __init__(
      self, 
      layers: Tuple[int, ...], 
      activation=relu, 
      activation_derivative=relu_derivative, 
      activation_exceptions: dict[int, Callable]={},
      activation_derivative_exceptions: dict[int, Callable]={},
      optimizer: Optimizer = Adam()
  ) -> None:
    self.optimizer = optimizer
    self.activation = activation
    self.activation_derivative = activation_derivative
    self.activation_exceptions = activation_exceptions
    self.activation_derivative_exceptions = activation_derivative_exceptions

    if (activation_exceptions.get(len(layers)-1, None)) is None:
      self.activation_exceptions[len(layers)-1] = linear

    if (activation_derivative_exceptions.get(len(layers)-1, None)) is None:
      self.activation_derivative_exceptions[len(layers)-1] = linear_derivative

    self.init_coefficients(layers)

    # These are used in training
    self.weight_gradient = None
    self.bias_gradient = None

  def init_coefficients(self, layers: Tuple[int, ...]) -> None:
    self.layers = layers
    num_layers = len(self.layers) - 1

    self.biases: np.ndarray = np.array([None] * num_layers) 
    self.weights: np.ndarray = np.array([None] * num_layers)

    for i in range(num_layers):
      max = math.sqrt(2/layers[i])
      min = -max
      self.biases[i] = config.rng.uniform(min,max,(layers[i+1], 1))
      self.weights[i] = config.rng.uniform(min,max,(layers[i+1], layers[i]))

  def feedforward_full(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_layers = len(self.layers) - 1
    activations: List = [None] * num_layers
    zs: List = [None] * num_layers
  
    last_activations = input
    for i, _ in enumerate(activations):
      zs[i], activations[i] = self.feedforward_layer(i, last_activations)
      last_activations = activations[i]

    return zs, activations

  def feedforward(self, input):
    return self.feedforward_full(input)[1][-1]

  def feedforward_layer(self, i: int, last_activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # z_{i} = w * a_{i-1} + b
    z = np.matmul(self.weights[i], last_activations) + self.biases[i]

    # Sometimes, the default activation function, self.activation,
    # will not always be the activation for every layer. For instance,
    # ReLu is not often the activation for the final layer since negative
    # results could not then be returned ever, so self.activation_exceptions
    # would have a value at key=final_layer_index with value=sigmoid
    
    # In activation_exceptions, i=0 corresponds to the first (input) layer
    
    activation_function = self.activation_exceptions.get(i+1, self.activation)
    return (z, activation_function(z))

  def loss(self, y_true, y_pred):
    n = y_true.shape[0]  # Number of samples

    # Reconstruction loss
    reconstruction_loss = np.sum(np.square(y_true - y_pred)) / n

    return (reconstruction_loss,)

  @staticmethod
  def graph_loss(losses, test_losses = []):
    sub = plt.subplots(2 if not len(test_losses) == 0 else 1, sharex=True)
    axs: Any = sub[1]
    if len(test_losses) == 0:
      axs = [axs]
    axs[0].plot(losses, "purple", label="Total Loss")
    axs[0].set(ylabel='Loss',
     title='Loss over time (Training Data)')
    axs[0].legend(loc="upper left")
    axs[0].grid()
    axs[0].semilogy()

    if test_losses:
      axs[1].plot(test_losses, "purple", label="Total Loss")
      axs[1].set(xlabel='Mini Batch', ylabel='Loss',
       title='Loss over time (Test Data)')
      axs[1].legend(loc="upper left")
      axs[1].grid()
      axs[1].semilogy() 

    plt.show()


  def train(self, _training_data: np.ndarray, max_epochs: int, batch_size:int=100, test_data: (np.ndarray|None)=None, learning_rate=0.05, graph=False, print_epochs=True) -> None:
    losses = []
    test_losses = []

    training_data = np.array(_training_data, copy=True)

    per_epoch = len(training_data) // batch_size

    # This is because, with a lot of inputs, we have a harder
    # time stabilizing gradient.
    adjusted_learning_rate = learning_rate / len(training_data[0][0])

    if per_epoch == 0:
      raise Exception("Batch Size greater than Data Set")
    for i in range(max_epochs):
      np.random.shuffle(training_data)
      for j in range(per_epoch):
        index = j * batch_size
        batch = training_data[index:index+batch_size]

        reconstruction_loss = self.training_step(batch, adjusted_learning_rate, print_epochs)[0]
        loss = reconstruction_loss
        losses.append(loss)

        test_loss = 0
        if not(test_data is None) and print_epochs:
          for index, element in enumerate(test_data):
            delta_test_reconstruction_loss = self.loss(test_data[index][1], self.feedforward(test_data[index][0]))[0]
            test_loss += delta_test_reconstruction_loss

          test_loss /= len(test_data)
          test_losses.append(test_loss)

        if print_epochs:
          print(f"Epoch {i}, Mini-Batch {j}: Loss = {loss}, Test Loss = {test_loss}")
    if graph:
      self.graph_loss(losses, test_losses)

  def init_gradients(self):
    if self.weight_gradient is None or self.bias_gradient is None:
      self.weight_gradient = np.empty(self.weights.shape, np.ndarray)
      self.bias_gradient = np.empty(self.biases.shape, np.ndarray)
    for i, _ in enumerate(self.weight_gradient):
      self.weight_gradient[i] = np.zeros(self.weights[i].shape)
      self.bias_gradient[i] = np.zeros(self.biases[i].shape)

  def training_step(self, batch, learning_rate, print_epochs):
    self.init_gradients()
    if (self.weight_gradient is None or self.bias_gradient is None):
      raise Exception("weight gradient not defined for some reason")
    reconstruction_loss = 0

    len_layers = len(self.layers)
    for _, data_point in enumerate(batch):
      z_values, activations = self.feedforward_full(data_point[0])

      # Partial Derivative of Loss with respect to the output activations
      dL_daL = (activations[-1] - data_point[1]) * (2/len(activations[-1]))
      if print_epochs:
        delta_reconstruction_loss = self.loss(data_point[1], activations[-1])[0]
        reconstruction_loss += delta_reconstruction_loss

      len_z = len(z_values)

      decoder_gradients_z = np.array([None] * len_z)

      decoder_gradients_z[-1] = dL_daL * self.activation_derivative_exceptions.get(len_layers-1, self.activation_derivative)(z_values[-1])

      last_index = len_z - 1

      for j in range(last_index, -1, -1):
        z_layer = z_values[j]
        if j != last_index:
          activation_derivative_func = self.activation_derivative_exceptions.get(j+1, self.activation_derivative)
          decoder_gradients_z[j] = np.matmul(
            self.weights[j+1].transpose(), decoder_gradients_z[j+1]
          ) * activation_derivative_func(z_layer)
          
        if j != 0:
          self.weight_gradient[j] += np.matmul(decoder_gradients_z[j], activations[j-1].transpose())
          self.bias_gradient[j] += decoder_gradients_z[j]

      self.weight_gradient[0] += np.matmul(decoder_gradients_z[0], data_point[0].transpose())
      self.bias_gradient[0] += decoder_gradients_z[0]

    self.weights -= learning_rate/len(batch) * self.optimizer.adjusted_weight_gradient(self.weight_gradient/len(batch[0]))
    self.biases -= learning_rate/len(batch) * self.optimizer.adjusted_bias_gradient(self.bias_gradient/len(batch[0]))
    return (reconstruction_loss/len(batch),)

