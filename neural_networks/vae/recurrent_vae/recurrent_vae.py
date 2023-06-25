from neural_networks.vae.normal_vae.normal_vae import VAE
import neural_networks.components.config as config
from typing import Tuple
from neural_networks.components.activations import *
from neural_networks.components.optimizer.adam import Adam
from neural_networks.components.optimizer.optimizer import Optimizer
import numpy as np
import numpy.typing as npt
import math
from typing import Callable

class RecurrentVAE(VAE):

  def __init__(
      self, 
      encoder_layers: Tuple[int, ...], 
      decoder_layers: Tuple[int, ...], 
      activation=leaky_relu, 
      activation_derivative=leaky_relu_derivative, 
      activation_exceptions: dict[int, Callable]={},
      activation_derivative_exceptions: dict[int, Callable]={},
      optimizer: Optimizer = Adam()
  ) -> None:
    self.hidden_state_size = encoder_layers[-1]
    super().__init__(encoder_layers, decoder_layers, activation, activation_derivative, activation_exceptions, activation_derivative_exceptions, optimizer)

  def init_coefficients(self, e_layers: Tuple[int, ...], d_layers: Tuple[int, ...]) -> None:
    self.encoder_layers = e_layers
    self.decoder_layers = d_layers
    self.latent_size = d_layers[0]

    length = len(e_layers) + len(d_layers) - 2

    self.biases: np.ndarray = np.empty(length, dtype=np.ndarray)
    self.weights: np.ndarray = np.empty(length, dtype=np.ndarray)

    index = 0

    # Input to first hidden layer (Includes the fact that
    # the hidden state is passed into the first hidden layer
    # in the encoder)
    max = math.sqrt(2 / e_layers[index])
    min = -max
    self.biases[index] = config.rng.uniform(min, max, (e_layers[index+1], 1))
    self.weights[index] = config.rng.uniform(min, max, (e_layers[index+1], e_layers[index]+self.hidden_state_size))
    index+=1

    # Encoder layers
    for _ in range(1, len(e_layers)-2):
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

    # Sample to Decoder (Includes the fact that 
    # the output of iteration n-1 is passed into 
    # the first hidden layer in decoder)
    self.biases[index] = config.rng.uniform(min, max, (d_layers[1], 1))
    self.weights[index] = config.rng.uniform(min, max, (d_layers[1], d_layers[0]+d_layers[-1]))
    index+=1

    # Decoder layers
    for _ in range(0, len(d_layers)-2):
      max = math.sqrt(2 / d_layers[index+1-len(e_layers)])
      min = -max
      self.biases[index] = config.rng.uniform(min, max, (d_layers[index+2-len(e_layers)], 1))
      self.weights[index] = config.rng.uniform(min, max, (d_layers[index+2-len(e_layers)], d_layers[index+1-len(e_layers)]))
      index+=1

  def get_time_seperated_data(self, input_data):
    """This function divides the input data into evenly sized vectors\
with one for each time step.
    """
    input_layer_size = self.encoder_layers[0]
    input_data_size = input_data[0].shape[0]

    if input_data_size % input_layer_size != 0:
      raise Exception("Input data cannot be divided evenly into input layer")
    return np.array([data_point.reshape(input_data_size//input_layer_size,input_layer_size,1) for data_point in input_data])

  def encode(self, input_value: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function takes an input vector and returns a \
latent space vector.

   :param input_value An (M, N) vector of floats, where N is \
the size of the input layer and M is the number of \
iterations in the recurrent network.
    """

    iterations = input_value.shape[0]
    latent_size = self.encoder_layers[-1]

    activations = np.empty([iterations, latent_size], dtype=np.ndarray)
    z_values = np.empty([iterations, latent_size], dtype=np.ndarray)
    epsilon = np.empty([iterations-1, self.hidden_state_size, 1], dtype=np.ndarray)

    last_activations = np.concatenate( (input_value[0], np.empty( (self.hidden_state_size, 1) )) )
    for iter in range(0, iterations):
      for layer in range(0, len(self.encoder_layers)-1):
        z_values[iter][layer], activations[iter][layer] = self.feedforward_layer(layer, last_activations)
        last_activations = activations[iter][layer]

      if iter != iterations-1:
        new_hidden_state , epsilon[iter] = self.gen(activations[iter][-1][0:self.hidden_state_size], activations[iter][-1][self.hidden_state_size:self.hidden_state_size*2])

        last_activations = np.concatenate( (input_value[iter+1], new_hidden_state) )
    parameters_count = len(activations[-1][-1])//2

    return (
      z_values,
      activations,
      activations[-1][-1][:parameters_count], 
      activations[-1][-1][parameters_count:parameters_count*2],
      epsilon
    )

  def decode(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """This function takes an (N, 1) vector representing the latent\
space representation.

    :param input
    """
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

