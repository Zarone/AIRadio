# AI Radio

## To Do
- Create Music Parser
- Create VAE
  - Start with basic (element-wise) loss function.
    - This is for testing, and for comparison later on.
- Create GAN
- Create Web UI
  - Probably going to be in NextJS or a similar framework.
- Create Latent Space Descent

## Model
- Create Recurrent Variational Auto Encoder
  - With music as input and output
- Train Generative Adversarial Network
  - Randomly pick a song, S<sub>0</sub>.
  - Get the latent space coordinate, P, for the song using the encoder.
  - Use the latent space coordinate as the parameters to the generator function G(P).
  - Use S<sub>0</sub> and G(P) as the two possible inputs to the discriminator.
  - Use the value from the discriminator to train VAE/Generator
- Train From User Input
  - Add a new neural network layer or two to the generator.
  - Create a large neural network, D(G(P)) = D(S), which takes in a whole song and returns a prediction for the user loss function.
    - This function starts as a copy of the discriminator with an additional layer or two.
  - Repeatedly:
    - Randomly pick a coordinate, P, from the latent space
    - Use the generator to create song, S = G(P).
    - Play song S.
    - Use current value of user controlled slider to determine loss value, L, for D(S)
      - Train the last layer of this network using this value.
      - (Note that when I say train, the data may just be added to some training queue so that we can use mini batches)
    - Use current value of user controlled slider to determine loss value, L, for generator G(P).
      - Train the last layer of this network using this value.
      - ∂L/∂w = ∂L/∂G<sub>i</sub> * ∂G<sub>i</sub>/∂w
        - For a given loss, L, a given weight, w, and a given moment in a song, G<sub>i</sub>.
        - ∂L/∂G<sub>i</sub> comes from D(S)
    - At some point, use the GAN network to descend the gradient of the latent space to create a final song