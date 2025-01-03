#!/bin/bash
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'
python -m fast_molopt.vae_train_prop --train data/bi_isolig/preprocessed/ \
                                     --vocab vocabs/bi_isolig-vocab.txt \
                                     --save_dir models/bi_isolig/ \
                                     --train_mode None
