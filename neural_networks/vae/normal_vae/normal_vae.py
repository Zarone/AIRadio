from neural_networks.components.encoder import Encoder

class VAE:


  def __init__(self, layers):
    """
    Parameters
    ----------
    layers 
      this defines the number of nodes in each activation layer (including input space and latent space)
    """
    encoder = Encoder(layers)
    #decoder = Decoder[layers.reverse()]