from neural_networks.vae.normal_vae.normal_vae import VAE
import neural_networks.components.config as config
from typing import Tuple
from neural_networks.components.activations import\
    leaky_relu, leaky_relu_derivative
from neural_networks.components.optimizer.adam import Adam
from neural_networks.components.optimizer.optimizer import Optimizer
import numpy as np
import math


class RecurrentVAE(VAE):

    def __init__(
        self,
        encoder_layers: Tuple[int, ...],
        decoder_layers: Tuple[int, ...],
        activation=leaky_relu,
        activation_derivative=leaky_relu_derivative,
        optimizer: Optimizer = Adam()
    ) -> None:
        self.hidden_state_size = encoder_layers[-1]
        super().__init__(
            encoder_layers,
            decoder_layers,
            activation,
            activation_derivative,
            optimizer
        )

    def init_coefficients(
        self,
        e_layers: Tuple[int, ...],
        d_layers: Tuple[int, ...]
    ) -> None:
        self.encoder_layers = e_layers
        self.decoder_layers = d_layers
        self.latent_size = d_layers[0]

        self.layers = e_layers[0:-1] + d_layers

        length = len(e_layers) + len(d_layers) - 2

        self.biases: np.ndarray = np.empty(length, dtype=np.ndarray)
        self.weights: np.ndarray = np.empty(length, dtype=np.ndarray)

        index = 0

        # Input to first hidden layer (Includes the fact that
        # the hidden state is passed into the first hidden layer
        # in the encoder)
        max = math.sqrt(2 / e_layers[index])
        min = -max
        self.biases[index] = config.rng.uniform(
            min, max, (e_layers[index+1], 1))
        self.weights[index] = config.rng.uniform(
            min, max, (e_layers[index+1], e_layers[index]+self.hidden_state_size))
        index += 1

        # Encoder layers
        for _ in range(1, len(e_layers)-2):
            max = math.sqrt(2 / e_layers[index])
            min = -max
            self.biases[index] = config.rng.uniform(
                min, max, (e_layers[index+1], 1))
            self.weights[index] = config.rng.uniform(
                min, max, (e_layers[index+1], e_layers[index]))
            index += 1

        max = math.sqrt(2 / e_layers[index])
        min = -max

        # Encoder to latent space
        self.biases[index] = config.rng.uniform(
            min, max, (e_layers[index+1]*2, 1))
        self.weights[index] = config.rng.uniform(
            min, max, (e_layers[index+1]*2, e_layers[index]))
        index += 1

        max = math.sqrt(2 / d_layers[0])
        min = -max

        # Sample to Decoder (Includes the fact that
        # the output of iteration n-1 is passed into
        # the first hidden layer in decoder)
        self.biases[index] = config.rng.uniform(min, max, (d_layers[1], 1))
        self.weights[index] = config.rng.uniform(
            min, max, (d_layers[1], d_layers[0]+d_layers[-1]))
        index += 1

        # Decoder layers
        for _ in range(0, len(d_layers)-2):
            max = math.sqrt(2 / d_layers[index+1-len(e_layers)])
            min = -max
            self.biases[index] = config.rng.uniform(
                min, max, (d_layers[index+2-len(e_layers)], 1))
            self.weights[index] = config.rng.uniform(
                min, max, (d_layers[index+2-len(e_layers)], d_layers[index+1-len(e_layers)]))
            index += 1

    def get_time_seperated_data(self, input_data):
        """This function divides the input data into evenly sized vectors\
    with one for each time step.
        """
        input_layer_size = self.encoder_layers[0]
        input_data_size = input_data[0].shape[0]

        assert input_data_size % input_layer_size == 0, "Input data cannot be divided evenly into input layer"

        return_array = np.empty((len(input_data),), dtype=np.ndarray)
        for i, data_point in enumerate(input_data):
            return_array[i] = data_point.reshape(
                data_point.shape[0]//input_layer_size, input_layer_size, 1)

        return return_array

    def _encode(self, input_value: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """This function takes an input vector and returns all \
    internally relevant variables after feedforward to latent space.

       :param input_value An (M, N) vector of floats, where N is \
    the size of the input layer and M is the number of \
    iterations in the recurrent network.
        """

        iterations: int = input_value.shape[0]

        # The number of layers in the encoder
        num_layers: int = len(self.encoder_layers)

        activations = np.empty([iterations, num_layers-1], dtype=np.ndarray)
        z_values = np.empty([iterations, num_layers-1], dtype=np.ndarray)
        epsilon = np.empty(
            [iterations-1, self.hidden_state_size, 1],
            dtype=np.float64
        )
        combined_inputs = np.empty(
            (iterations, ),
            dtype=np.ndarray
        )

        last_activations = np.concatenate(
            (input_value[0], np.empty((self.hidden_state_size, 1))))
        combined_inputs[0] = last_activations
        for iter in range(0, iterations):
            for layer in range(0, len(self.encoder_layers)-1):
                z_values[iter][layer], activations[iter][layer] = self.feedforward_layer(
                    layer, last_activations)
                last_activations = activations[iter][layer]

            if iter != iterations-1:
                new_hidden_state, epsilon[iter] = \
                    self._gen(
                        activations[iter][-1][0:self.hidden_state_size], activations[iter][-1][self.hidden_state_size:self.hidden_state_size*2]
                )

                last_activations = np.concatenate(
                    (input_value[iter+1], new_hidden_state))
                combined_inputs[iter+1] = last_activations
        parameters_count = len(activations[-1][-1])//2

        return (
            z_values,
            activations,
            activations[-1][-1][:parameters_count],
            activations[-1][-1][parameters_count:parameters_count*2],
            combined_inputs,
            epsilon
        )

    def encode(self, input_value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """This function takes an input vector and returns mu and \
    log variance for the latent space.

       :param input_value An (M, N) vector of floats, where N is \
    the size of the input layer and M is the number of \
    iterations in the recurrent network.
        """
        _, _, mu, logvar, _, _ = self._encode(input_value)
        return (mu, logvar)

    def _decode(self, input_value: np.ndarray, iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """This function takes an (N, 1) vector representing the latent\
    space representation and returns all related z and activation values.

        :param input_value An (N, 1) vector representing the latent\
    space representation.
        """
        assert len(
            input_value.shape) == 2 and input_value.shape[1] == 1, f"Expected (N, 1) sized input, received {input_value.shape}"

        num_layers = len(self.decoder_layers)-1
        output_layer_size = self.decoder_layers[-1]

        activations: np.ndarray = np.empty(
            [iterations, num_layers], dtype=np.ndarray)
        z_values: np.ndarray = np.empty(
            [iterations, num_layers], dtype=np.ndarray)
        last_output = np.zeros((output_layer_size, 1))
        last_activations: np.ndarray = np.concatenate(
            (input_value, last_output))

        for i in range(iterations):
            for j in range(0, num_layers):
                coef_index = j+len(self.encoder_layers)-1
                z_values[i][j], activations[i][j] = self.feedforward_layer(
                    coef_index, last_activations)
                last_activations = activations[i][j]
            last_activations = np.concatenate((input_value, last_activations))

        return (z_values, activations)

    def decode(self, input_value: np.ndarray, iterations: int) -> np.ndarray:
        """This function takes an (N, 1) vector representing the latent\
    space representation and returns an M-length array of N-length arrays of floats\
    where M is the number of iterations.

        :param input_value An (N, 1) vector representing the latent\
    space representation.
        """
        return self._decode(input_value, iterations)[1][:, -1]

    def train(
        self,
        _training_data: np.ndarray,
        max_epochs: int,
        batch_size: int = 100,
        test_data: (np.ndarray | None) = None,
        learning_rate=0.05,
        graph=False,
        print_epochs=True
    ) -> None:
        """This function trains the neural networks on a dataset variable.

        :param _training_data A numpy array of length N where each element is a (P,Q,1) \
    shaped numpy array. N is the number of training examples, P is the number of iterations, \
    and Q is the length of the input layer.
        """

        assert len(
            _training_data.shape) == 1, f"Expected training data with shape (N, ), but got {_training_data.shape}"
        assert len(
            _training_data[0].shape) == 3, f"Expected training data point with shape (P,Q,1) but got {_training_data[0].shape}"

        losses = []
        kl_losses = []
        reconstruction_losses = []

        test_losses = []
        test_kl_losses = []
        test_reconstruction_losses = []

        training_data = np.array(_training_data, copy=True)
        per_epoch = len(training_data) // batch_size

        assert per_epoch != 0, "Batch Size greater than Data Set"

        for i in range(max_epochs):
            np.random.shuffle(training_data)
            for j in range(per_epoch):
                index = j * batch_size
                batch = training_data[index:index+batch_size]

                reconstruction_loss, kl_loss = self.training_step(
                    batch, learning_rate, print_epochs)

                kl_losses.append(kl_loss)
                reconstruction_losses.append(reconstruction_loss)
                loss = kl_loss+reconstruction_loss
                losses.append(loss)

                test_loss = 0
                if not (test_data is None) and print_epochs:
                    test_reconstruction_loss = 0
                    test_kl_loss = 0
                    for index, element in enumerate(test_data):
                        mu, logvar = self.encode(element)
                        generated = self.gen(mu, logvar)
                        delta_test_reconstruction_loss, delta_test_kl_loss = self.loss(
                            element, self.decode(generated)[1][-1], mu, logvar)
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
                    print(
                        f"Epoch {i}, Mini-Batch {j}: Loss = {loss}, Test Loss = {test_loss}")
                    # print(f"Epoch {i}, Mini-Batch {j}: KL Loss = {kl_loss}, Reconstruction Loss = {reconstruction_loss}")
        if graph:
            self.graph_loss(losses, reconstruction_losses, kl_losses,
                            test_losses, test_reconstruction_losses, test_kl_losses)

    def training_step(self, batch, learning_rate, print_epochs):
        self.init_gradients()

        assert (not self.weight_gradient is None and not self.bias_gradient is None), "Weight gradient not defined for some reason"

        reconstruction_loss = 0
        kl_loss = 0

        # Beta affects the relative importance of kl_loss
        # with respect to reconstruction_loss in calculating
        # the gradient.
        Beta = 1

        for i, data_point in enumerate(batch):
            print(f"data_point {i}")

            z1, a1, mu, log_variance, decoder_inputs, epsilon_encoder = self._encode(
                data_point)
            generated, epsilon = self._gen(mu, log_variance)
            z2, a2 = self._decode(generated, len(data_point))

            # This is just for readability later on
            z_values = np.concatenate((z1.T, z2.T)).T
            activation_values = np.concatenate((a1.T, a2.T)).T

            mu_logvar_values = z_values[:, len(
                self.encoder_layers)-2][0:self.decoder_layers[0]]
            mu_values = np.array([x[0:self.decoder_layers[0]]
                                 for x in mu_logvar_values])
            logvar_values = np.array(
                [x[self.decoder_layers[0]:] for x in mu_logvar_values])

            # This is because it simplifies backpropagation calculations
            z_values[:, len(self.encoder_layers)-2] = decoder_inputs[:]
            activation_values[:, len(
                self.encoder_layers)-2] = decoder_inputs[:]

            # for j, time_step in enumerate(data_point):
            j = len(data_point)-1
            time_step = data_point[-1]
            print(f"time step {j}")

            # Backpropagate through decoder
            # Find starting dL_dz, set last_dL_dz to this
            # For each time step
            # For each layer to first hidden layer in decoder
            # weight_gradient, output_dL_dz += backpropagate(layer, last_dL_dz, activations, ...)
            # where output_dL_dz is the dL_dz for the output of the last timestep
            # set last_dL_dz to output_dL_dz
            # Find dL_dmu, dL_dlogvar like before. Get dL_dz from this
            # set last_dL_dz to that dL_dz from dL_dmu and dL_dlogar
            # For each time step
            # For each layer to first hidden layer in encoder
            # weight_gradient, output_dL_dz += backpropagate(layer, last_dL_dz, activations, ...)
            # where output_dL_dz is the dL_dz for the output of the last timestep
            # set last_dL_dz to output_dL_dz

            dL_daL = np.array((activation_values[j][-1] - time_step) * (
                2/len(activation_values[j][-1])), dtype=np.float64)

            last_layer_index = len(self.weights) - 1
            decoder_stop_layer = len(self.encoder_layers) - 2

            # Backpropagates through time, from the output layer
            # to the start of the decoder.
            last_dL_dz = self.backpropagate_through_time(
                j,
                last_layer_index,  # first layer (inclusive)
                decoder_stop_layer,  # last layer (exclusive)
                dL_daL,
                z_values,
                activation_values
            )

            # ∂L/∂mu = ∂L/∂z_n * ∂z_n/∂a_(n-1) * ∂a_(n-1)/∂mu + ∂L/∂D * ∂D/∂mu
            #        = ∂L/∂z_n * w_n           * 1            + 1     * mu/N
            #        = ∂L/∂z_n * w_n + mu/N
            dL_dmu = last_dL_dz + Beta*mu/self.latent_size

            # ∂L/∂logvar = ∂L/∂z_n * ∂z_n/∂a_(n-1) * ∂a_(n-1)/∂logvar                + ∂L/∂D   * ∂D/∂logvar
            #            = ∂L/∂z_n * w_n           * epsilon / 2*np.exp(logvar/2)    + 1       * (1-np.exp(logvar))/2N
            #            = ∂L/∂z_n * w_n * epsilon/2*np.exp(logvar/2) - (1-np.exp(logvar))/2N
            dL_dlogvar = \
                last_dL_dz \
                * epsilon/2*np.exp(log_variance/2) \
                - Beta*(1-np.exp(log_variance))/self.latent_size/2

            last_dL_dz = np.concatenate((dL_dmu, dL_dlogvar), axis=0)

            last_dL_dz = self.backpropagate_through_time(
                j,
                decoder_stop_layer,  # first layer (inclusive)
                -1,  # last layer (exclusive)
                last_dL_dz,
                z_values,
                activation_values,
                iter_input=z_values[:, len(self.encoder_layers)-2],
                Beta=Beta,
                mu=mu_values,
                log_variance=logvar_values,
                epsilon=epsilon_encoder
            )

        return (0, 0)

    def backpropagate_through_time(
        self,
        time_start: int,
        start_layer: int,
        end_layer: int,
        start_dL_dz: np.ndarray,
        z_values: np.ndarray,
        activation_values: np.ndarray,
        iter_input: (np.ndarray | None) = None,
        Beta=None,
        mu=None,
        log_variance=None,
        epsilon=None
    ) -> np.ndarray:
        """Backpropagates through time.

        :param time_start The time step which this iteration \
    of backporpagation starts with.
        :param start_layer This is the layer which backpropagation\
    starts with. Inclusive.
        :param end_layer This is the layer which backpropagation\
    end with. Exclusive.
        :param last_dL_dz This is the partial derivative of loss\
    with respect to the z values from the previous layer.
        """
        last_dL_dz = start_dL_dz

        assert (not self.weight_gradient is None and not self.bias_gradient is None), "Weight gradient not defined for some reason"

        for i in range(time_start, -1, -1):
            print(f"backprop time step {i}")

            for layer in range(start_layer, end_layer, -1):
                print(f"layer {layer}")
                z_layer = z_values[i][layer-1]
                a_layer = activation_values[i][layer-1]
                if layer == 0:
                    assert iter_input is not None,\
                        "iter_input expected but not provided"
                    z_layer = iter_input[i-1]
                    a_layer = iter_input[i-1]
                delta_weight_gradient, delta_bias_gradient, last_dL_dz =\
                    self.backpropagate_layer(
                        layer,
                        last_dL_dz,
                        z_layer,
                        a_layer
                    )
                self.weight_gradient[layer] += delta_weight_gradient
                self.bias_gradient[layer] += delta_bias_gradient

            # If the backpropagation is through the decoder
            if end_layer > -1:
                # If we keep going back through time
                # then we're going to need to backpropagate
                # through only the last output, which is in range
                # [hidden_state_size to the end) and if we aren't
                # then we backpropagate through the hidden state
                # which is in range [0, hidden_state_size).
                if i != 0:
                    last_dL_dz = last_dL_dz[self.hidden_state_size:]
                else:
                    last_dL_dz = last_dL_dz[:self.hidden_state_size]
            elif end_layer == -1:
                assert (Beta is not None)\
                    and (mu is not None)\
                    and (log_variance is not None)\
                    and (epsilon is not None),\
                    "Beta, mu, log variance, and epsilon not passed to method"

                # We only need to include the derivatives with
                # respect to state, not input
                last_dL_dz = last_dL_dz[self.encoder_layers[0]:]
                dL_dmu = last_dL_dz + Beta*mu[i]/self.latent_size
                dL_dlogvar = \
                    last_dL_dz \
                    * epsilon[i-1]/2*np.exp(log_variance[i]/2) \
                    - Beta*(1-np.exp(log_variance[i]))/self.latent_size/2
                last_dL_dz = np.concatenate(
                    (dL_dmu, dL_dlogvar), axis=0, dtype=np.float64)

        return last_dL_dz

    def backpropagate_layer(
        self,
        layer: int,
        dL_dz: np.ndarray,
        z_layer: np.ndarray,
        a_layer: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        activation_derivative_func = self.activation_derivative

        local_weights = self.weights[layer]
        output_dL_dz = np.matmul(
            local_weights.T, dL_dz
        ) * activation_derivative_func(z_layer)

        weight_gradient = np.matmul(dL_dz, a_layer.T)
        bias_gradient = dL_dz

        return (weight_gradient, bias_gradient, output_dL_dz)
