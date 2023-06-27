from tools.tester import Tester
from neural_networks.components.component_test import component_test
from neural_networks.vae.normal_vae.normal_vae_test import normal_vae_test 
from neural_networks.vae.recurrent_vae.rvae_test import rvae_test
from neural_networks.autoencoder.autoencoder_tester import autoencoder_test

component_test(Tester)
autoencoder_test(Tester)
normal_vae_test(Tester)
rvae_test(Tester)
