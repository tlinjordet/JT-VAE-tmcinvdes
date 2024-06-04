# Accelerated Training of Junction Tree VAE

The MOSES dataset can be downloaded from https://github.com/molecularsets/moses.

## Deriving Vocabulary

If you are running our code on a new dataset, you need to compute the vocabulary from your dataset.
To perform tree decomposition over a set of molecules, run

```
cd fast_jtnn
python mol_tree.py -i ./../../data/train.txt -v ./../../data/vocab.txt
```

This gives you the vocabulary of cluster labels over the dataset `train.txt`.

## Training

Step 1: Preprocess the data:

```
python preprocess.py --train ../data/train.txt --split 100 --jobs 40 --output ./moses-processed
```

This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files. We suggest you to use small value for `--split` if you are working with smaller datasets.

Step 2: Train VAE model with KL annealing.

```
python vae_train.py --train moses-processed --vocab ../data/vocab.txt --save_dir vae_model/
```

Default Options:

`--beta 0` means to set KL regularization weight (beta) initially to be zero.

`--warmup 500` means that beta will not increase within first 500 training steps. It is recommended as using large KL regularization (large beta) in the beginning of training is harmful for model performance.

`--step_beta 0.002 --kl_anneal_iter 1000` means beta will increase by 0.002 every 1000 training steps (batch updates). You should observe that the KL will decrease as beta increases.

`--max_beta 1.0 ` sets the maximum value of beta to be 1.0.

`--save_dir vae_model`: the model will be saved in vae_model/

Please note that this is not necessarily the best annealing strategy. You are welcomed to adjust these parameters. Additionally, the parameters are dataset size dependant.
A larger dataset means that the number of iterations will be higher before the whole dataset is run trough once (1 epoch).
