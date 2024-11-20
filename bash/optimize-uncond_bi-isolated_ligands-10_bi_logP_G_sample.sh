#!/bin/bash
python -m fast_molopt.optimize --input_dir_path data/uncond_bi_ligand_desc/uncond_bi-min10k-labeled-included-isolated_ligands-sampled_for_cond_bi-input_columns.csv  \
                               --vocab_path vocabs/isolated_ligands_8b_uncond_bi-vocab.txt \
                               --model_path models/uncond_bi_ligand_desc/model.epoch-149 \
                               --denticity bidentate \
                               --train_mode None \
                               --labeling isolated_ligands
