# FastJTNNpy3 : Junction Tree Variational Autoencoder for Molecular Graph Generation

This repo is a fork of the repo : "Python 3 Version of Fast Junction Tree Variational Autoencoder for Molecular Graph Generation (ICML 2018)"


<img src="https://github.com/Bibyutatsu/FastJTNNpy3/blob/master/Old/paradigm.png" width="600">

Implementation of our Junction Tree Variational Autoencoder [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364)

# Requirements
The version of RDKit is very important. For newer versions of RDKit the model does not work!
The tree decomposition has kekulization errors if this is the case.
The requirements.yml file is an export of a working conda environment that can run this model.

# Quick Start

## Code for Accelerated Training
This repository contains the Python 3 implementation of the new Fast Junction Tree Variational Autoencoder code.

* `fast_molvae/` contains codes for VAE training. Please refer to `fast_molvae/README.md` for details.
* `fast_jtnn/` contains codes for model implementation.
* `fast_molopt/` contains codes for molecule optimisation using a JTpropVAE which is the same as JTVAE but trains with molecule properties as well.
