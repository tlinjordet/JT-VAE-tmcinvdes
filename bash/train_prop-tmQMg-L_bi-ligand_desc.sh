#!/bin/bash
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'
python -m fast_molopt.vae_train_prop --train data/tmQMg-L_bi-ligand_desc/preprocessed/ \
                                     --vocab vocabs/tmQMg-L_bi-vocab.txt \
                                     --save_dir models/tmQMg-L_bi-ligand_desc/ \
                                     --train_mode None \
                                     --batch_size 4
