# AI Radio

## Model
- Create Recurrent Variational Auto Encoder
  - With music as input and output
- Train Generative Adversarial Network
  - Randomly pick a song, S<sub>0</sub>.
  - Get the latent space coordinate, P, for the song using the encoder.
  - Use the latent space coordinate as the parameters to the generator function G\(P\). - Use S<sub>0</sub> and G\(P\) as the two possible inputs to the discriminator.
  - Use the value from the discriminator to train VAE/Generator
  - Generate a full latent space distribution using the training data
    - $\mu = \frac{\sum\mu_i}{n}$

    - $\sigma = \sqrt{\sum{\sigma_i^2}}$
- Train From User Input
  - Add a new neural network layer or two to the generator.
  - Create a large neural network, D(G\(P\)) = D(S), which takes in a whole song and returns a prediction for the user loss function.
    - This function starts as a copy of the discriminator with an additional layer or two.
  - Repeatedly:
    - Randomly pick a coordinate, P, from the latent space
    - Use the generator to create song, S = G\(P\).
    - Play song S.
    - Use current value of user controlled slider to determine loss value, L, for D(S)
      - Train the last layer of this network using this value.
      - (Note that when I say train, the data may just be added to some training queue so that we can use mini batches)
    - Use current value of user controlled slider to determine loss value, L, for generator G\(P\).
      - Train the last layer of this network using this value.
      - ∂L/∂w = ∂L/∂G<sub>i</sub> * ∂G<sub>i</sub>/∂w
        - For a given loss, L, a given weight, w, and a given moment in a song, G<sub>i</sub>.
        - ∂L/∂G<sub>i</sub> comes from D(S)
    - At some point, use the GAN network to descend the gradient of the latent space to create a final song

## Where this project is at:

- As of the last update, the Variational Auto-Encoder is written with compatibility with Recurrence.
    ```python
    import audio_parsing.audio_parsing as audio
    from neural_networks.components.recurrent import Recurrent
    from neural_networks.vae.vae import VAE

    network: VAE = VAE(
        encoder_args=dict(
            input_size=5, # The number of input values for each time step
            input_layers=(9, 4), # The number of nodes in each layer before the recurrence
            output_layers=(4, 6) # The number of nodes in each layer after the recurrence
        ),
        decoder_args=dict(
            # The number of input values to the decoder
            # Half the number of encoder outputs because half the decoder outputs are for variance
            input_size=3,

            input_layers=(7, 4), # The number of nodes in each layer before the recurrence
            output_layers=(4, 5) # The number of nodes in each layer after the recurrence
        ),
        latent_size=3, # The number of nodes in the latent space (encoding layer) of auto encoder
        sub_network=Recurrent # The type of network that will be used for encoder and decoder
    )

    # Takes array of inputs and seperates them into timesteps
    # Not neccessary if `sounds` is already seperated by time step
    time_separated_sounds = network.get_time_seperated_data(sounds)

    network.train(
        time_separated_sounds,
        batch_size=5,
        max_epochs=20000,
        graph=True,
        learning_rate=0.001,
        time_separated_input=True, # Expects input data to be seperated by time step
        time_separated_output=True # Expects output data to be seperated by time step
    )
    ```
- In addition, I also wrote some functions for easy data compression using autoencoding
    ```python
    import audio_parsing.audio_parsing as audio
    from compression.compression import (
        train_compressor,
        COMPRESSION_1_INFO,
        compress,
        decompress
    )

    AMPLITUDE_SCALE = 1
    NUM_AMPLITUDES = 15
    NUM_FILES = 5

    sounds, names = audio.get_raw_data(NUM_FILES, NUM_AMPLITUDES, AMPLITUDE_SCALE)


    # This is just a way to test the sound data
    song = audio.play_random_sound(sounds, names, AMPLITUDE_SCALE)

    train_compressor(sounds, COMPRESSION_1_INFO, 10000)

    compressed, ae1 = compress(song, COMPRESSION_1_INFO)
    print(f"Current Size of Song: {len(compressed)}")

    decompressed = decompress(ae1, compressed, COMPRESSION_1_INFO)
    print(f"Current Size of Song: {len(decompressed)}")

    audio.play_audio(decompressed, AMPLITUDE_SCALE)

    audio.plot_audio_comparison(song, decompressed)
    ```

## To Do
- Create GAN
- Create Web UI
  - Probably going to be in NextJS or a similar framework.
- Create Latent Space Descent

