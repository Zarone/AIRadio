from neural_networks.components.base import BaseNetwork
import numpy as np
import neural_networks.components.config as config
from neural_networks.components.activations import relu, relu_derivative
from typing import Tuple, Callable, Any
import matplotlib.pyplot as plt

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
    # Reconstruction loss
    reconstruction_loss = np.mean(np.square(y_true - y_pred)[0], axis=-1)

    print("KL Divergence per Latent Attribute")
    print(-0.5*(1 + log_var - np.square(mu) - np.exp(log_var)))

    # Regularization term - KL divergence
    kl_loss = -0.5 * np.mean( (1 + log_var - np.square(mu) - np.exp(log_var)) )

    print("kl_loss")
    print(kl_loss)

    # Total loss
    # total_loss = reconstruction_loss + kl_loss

    return reconstruction_loss, kl_loss 


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

    # print(f"layer = {i}")
    # print("last_activations")
    # print(last_activations)
    # print("weights")
    # print(self.weights[i])
    # print("biases")
    # print(self.biases[i])

    # print(f"z[{i}]")

    # z_{i} = w * a_{i-1} + b
    z_values[i] = np.matmul(self.weights[i], last_activations) + self.biases[i]

    # print(z_values[i])

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
    epsilon = (np.random.randn(len(mu)).reshape(-1, 1))/10
    print("epsilon")
    print(epsilon)
    print("log variance")
    print(log_variance)
    print("standard deviation")
    print(np.exp(0.5*log_variance))
    z = mu + np.exp(0.5 * log_variance) * epsilon
    return (z, epsilon)

  @staticmethod
  def graph_loss(losses, reconstruction_losses, kl_losses):
    sub = plt.subplots(2, sharex=True)
    axs: Any = sub[1]
    # axs[0].plot(losses, "purple", label="Total Loss")
    axs[0].plot(kl_losses, "red", label="KL Divergence")
    # axs[0].plot(reconstruction_losses, "blue", label="Reconstruction Loss")
    axs[0].set(ylabel='Loss',
     title='Loss over time (Training Data)')
    axs[0].legend(loc="upper left")
    # axs[1].plot(losses, "purple", label="Total Loss")
    axs[1].plot(kl_losses, "red", label="KL Divergence")
    # axs[1].plot(reconstruction_losses, "blue", label="Reconstruction Loss")
    axs[1].set(xlabel='Mini Batch', ylabel='Loss',
     title='Loss over time (Test Data)')
    axs[1].legend(loc="upper left")
    axs[0].grid()
    axs[1].grid()
    plt.show()


  def train(self, _training_data: np.ndarray, max_epochs: int, batch_size:int=100, test_data: (np.ndarray|None)=None, learning_rate=0.05, graph=False) -> None:
    losses = []
    kl_losses = []
    reconstruction_losses = []
    training_data = np.array(_training_data, copy=True)
    per_epoch = len(training_data) // batch_size
    for i in range(max_epochs):
      np.random.shuffle(training_data)
      for j in range(per_epoch):

        print(f"NEW MINIBATCH {i*per_epoch + j}")
        print("")
        print("")

        index = j * batch_size
        batch = training_data[index:index+batch_size]
        reconstruction_loss, kl_loss = self.training_step(batch, learning_rate)
        kl_losses.append(kl_loss)
        reconstruction_losses.append(reconstruction_loss)
        loss = kl_loss+reconstruction_loss
        losses.append(loss)
        print(f"Epoch {i}, Mini-Batch {j}: Loss = {loss}")
      if test_data:
        raise NotImplementedError("Testing not implemented")
    if graph:
      self.graph_loss(losses, reconstruction_losses, kl_losses)

  def training_step(self, batch, learning_rate):
    # This gradient is added to for each data point in the batch
    weight_gradient = np.empty(self.weights.shape, np.ndarray)
    bias_gradient = np.empty(self.biases.shape, np.ndarray)
    dL_dlogvar_all = np.zeros((self.latent_size, 1))
    dL_dmu_all = np.zeros((self.latent_size, 1))
    reconstruction_loss = 0
    kl_loss = 0

    for i, _ in enumerate(weight_gradient):
      weight_gradient[i] = np.zeros(self.weights[i].shape)
      bias_gradient[i] = np.zeros(self.biases[i].shape)

    for i, data_point in enumerate(batch):
      print(f"NEW DATA POINT {i}")
      print("")
      print("")

      z1, a1, mu, log_variance = self.encode(data_point)
      generated, epsilon = self.gen(mu, log_variance)
      z2, a2 = self.decode(generated)

      print("z1")
      print(z1)
      # print("generated latent space")
      # print(generated)
      # print("z2")
      # print(z2)

      # These are needed for some gradient calculations
      z_values = np.hstack((z1, z2))
      activations = np.hstack((a1, a2))
      activations[len(self.encoder_layers)-2] = generated

      # Partial Derivative of Loss with respect to the output activations
      dL_daL = (activations[-1] - data_point) * (2/len(activations[-1]))
      delta_reconstruction_loss, delta_kl_loss = self.vae_loss(data_point, activations[-1], mu, log_variance)
      reconstruction_loss += delta_reconstruction_loss
      kl_loss += delta_kl_loss

      # Loss Gradients with respect to z, for just the decoder
      decoder_gradients_z = np.empty((len(z_values)), np.ndarray)
      decoder_gradients_z[-1] = dL_daL * self.activation_derivative(z_values[-1])

      last_index = len(z_values) - 1
      first_index = len(self.encoder_layers) - 2

      # # Backpropagate through Decoder
      # for j in range(last_index, first_index, -1):
        # if j != last_index:
          # decoder_gradients_z[j] = np.matmul(
              # self.weights[j+1].transpose(), decoder_gradients_z[j+1]
            # ) * self.activation_derivative(z_values[j])
        # weight_gradient[j] += np.matmul(decoder_gradients_z[j], activations[j-1].transpose())
        # bias_gradient[j] += decoder_gradients_z[j]


      # ∂L/∂mu = ∂L/∂z_n * ∂z_n/∂a_(n-1) * ∂a_(n-1)/∂mu + ∂L/∂D * ∂D/∂mu
      #        = ∂L/∂z_n * w_n           * 1            + 1    * mu/N
      #        = ∂L/∂z_n * w_n + mu/N
      # dL_dmu = np.matmul(self.weights[first_index+1].transpose(), decoder_gradients_z[first_index+1]) + mu/self.latent_size
      dL_dmu = mu/self.latent_size
      dL_dmu_all += dL_dmu
      print("dL_dmu")
      print(dL_dmu)
      print("dL_dmu_all")
      print(dL_dmu_all)

      # ∂L/∂logvar = ∂L/∂z_n * ∂z_n/∂a_(n-1) * ∂a_(n-1)/∂logvar                + ∂L/∂D   * ∂D/∂logvar
      #            = ∂L/∂z_n * w_n           * epsilon/2*np.exp(logvar/2)      + 1       * (1-np.exp(logvar))/2N
      #            = ∂L/∂z_n * w_n * epsilon/2*np.exp(logvar/2) - (1-np.exp(logvar))/2N

      dL_dlogvar = -(1-np.exp(log_variance))/self.latent_size/2
      dL_dlogvar_all += dL_dlogvar

      # This section is here for a very specific reason. 
      # ∂D/∂logvar = -(1-e^logvar)/2N
      # This means that if logvar becomes extremely negative
      # the training will nearly stop because ∂D/∂logvar
      # will become constant. Overcorrection on this variable is sort
      # of dangerous, so this normalizes it. Seems like a weird hypothetical
      # but I had some trouble and this became necessary.
      max_grad_norm = np.linalg.norm(log_variance)
      grad_norm = np.linalg.norm(dL_dlogvar)
      if grad_norm > max_grad_norm:
          dL_dlogvar = dL_dlogvar * (max_grad_norm / grad_norm)

      # dL_dlogvar = \
        # np.matmul( self.weights[first_index+1].transpose(), decoder_gradients_z[first_index+1] ) \
        # * epsilon/2*np.exp(log_variance/2) \
        # - (1-np.exp(log_variance))/self.latent_size/2

      print("dL_dlogvar")
      print(dL_dlogvar)
      print("dL_dlogvar_all")
      print(dL_dlogvar_all)

      last_index = first_index
      first_index = -1

      decoder_gradients_z[last_index] = np.vstack((dL_dmu, dL_dlogvar))
      print(f"decoder_gradients_z[{last_index}]")
      print(decoder_gradients_z[last_index])

      # Backpropagate through Encoder
      for j in range(last_index, first_index, -1):
        print(f"j = {j}")
        if j != last_index:
          decoder_gradients_z[j] = np.matmul(
              self.weights[j+1].transpose(), decoder_gradients_z[j+1]
            ) * self.activation_derivative(z_values[j])
          print(f"decoder_gradients_z[{j}]")
          print(decoder_gradients_z[j])
        if j != 0:
          # print("decoder_gradients_z[j]")
          # print(decoder_gradients_z[j])
          # print("activations[j-1].transpose()")
          # print(activations[j-1].transpose())
          weight_gradient[j] += np.matmul(decoder_gradients_z[j], activations[j-1].transpose())
          bias_gradient[j] += decoder_gradients_z[j]
          # print(f"weight_gradient")
          # print(weight_gradient)
          # print(f"bias_gradient")
          # print(bias_gradient)

        # print("decoder_gradients_z[0]")
        # print(decoder_gradients_z[0])
        # print("data_point.transpose()")
        # print(data_point.transpose())
        weight_gradient[j] += np.matmul(decoder_gradients_z[j], data_point.transpose())
        bias_gradient[j] += decoder_gradients_z[j]
        print(f"new weight_gradient[j]")
        print(np.matmul(decoder_gradients_z[j], data_point.transpose()))
        # print(f"new bias_gradient[j]")
        # print(decoder_gradients_z[j])
        
        # print(f"weight_gradient")
        # print(weight_gradient)
        # print(f"bias_gradient")
        # print(bias_gradient)

    print("weights before")
    print(self.weights)
    self.weights -= learning_rate * weight_gradient/len(batch)
    print("weights after")
    print(self.weights)
    self.biases -= learning_rate * bias_gradient/len(batch)
    return reconstruction_loss/len(batch), kl_loss/len(batch)

