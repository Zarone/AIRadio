from tools.tester import Tester
from neural_networks.components.component_test import component_test
from neural_networks.autoencoder.autoencoder_tester import autoencoder_test
from neural_networks.vae.vae_test import vae_test

component_test(Tester)
autoencoder_test(Tester)
vae_test(Tester)
