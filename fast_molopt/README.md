# Constrained Molecule Optimization

## Important notes

The original property JT-VAE was not compatible with the rest of the code. I rewrote it to follow the datahandling and the updated "fast" code.
It was not clear to me how the processing of SMILES to moltrees was refactored.
Instead of rewriting the existing datautils.py i created a separate file datautils_prop.py. In this way i avoided dealing with one script having to handle both the data for the regular JT-VAE without properties as training input and the data for JT_prop_vae where properties were given as input as well.

## Training of conditional JT-VAE

Is done in the same way as the unconditional JT-VAE (see file fast_molvae/README.md)

## Optimization in latent space

To do directional optimization in latent space run optimize.py.
This script need a input csv file with encoded SMILES and their corresponding DFT labeled properties.
It also needs a trained model and vocabulary.

An example of running the optimization is:

```
python -u fast_molopt/optimize.py input_dir_path data/example_prompt_ligands.csv --vocab_path vocab.txt --cutoff 0.2 --lr 1.5 --model_path $model
```
