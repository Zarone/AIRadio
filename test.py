from tools.tester import Tester
from neural_networks.components.component_test import component_test
from neural_networks.vae.nn_VAE_test import nn_VAE_test

component_test(Tester)
nn_VAE_test(Tester)
