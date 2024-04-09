# Constrained Molecule Optimization

## Important notes
The original property JT-VAE was not compatible with the rest of the code. I rewrote it to follow the datahandling and the updated "fast" code.
It was not clear to me how the processing of SMILES to moltrees was refactored.
Instead of rewriting the existing datautils.py i created a separate file datautils_prop.py. In this way i avoided dealing with one script having to handle both the data for the regular JT-VAE without properties as training input and the data for JT_prop_vae where properties were given as input as well.

**IMPORTANT**
Around line 171 of datautils_prop.py there is a line related to preprocessing the JTMPN layer. This line should be commented out when using optimize.py! There is something different about how the data is prepared when using the model for conditional generation in optimize.py and this jtmpn_holder throws an exception. However, The jtmpn_holder is not used in optimize.py!
Therefore, a super dirty hack is to comment out the jtmpn_holder line and replace "jtmpn_holder" in the return statement with None. Then optimize.py will work.


## Train conditional JT-VAE

The difference form the regular training, is now we need to supply a file with the same length as the training file, which contains the properties for each molecule.

To do training run the following scripts in a bash submission:

```
python -u ../fast_jtnn/mol_tree.py -i $dataset -v $vocab
python -u preprocess_prop.py --train $dataset --split 10 --jobs 8 --output $output --prop_path $prop_path
python -u vae_train_prop.py --train $output --vocab $vocab --save_dir $save_dir
```

## Optimization in latent space


An example of running the optimization is:

```
python -u optimize.py --training_path $dataset --vocab_path $vocab --cutoff 0.2 --lr 2 --model_path $model --prop_path $prop_path
```
