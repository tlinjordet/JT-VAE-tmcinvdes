#!/bin/bash
python -m fast_molopt.optimize --input_dir_path data/uncond_mono_ligand_desc/uncond_mono-min15k-labeled-included-isolated_ligands-sampled_for_cond_mono-input_columns.csv  \
                               --vocab_path vocabs/isolated_ligands_8b_uncond_mono-vocab.txt \
                               --model_path models/uncond_mono_ligand_desc/model.epoch-149 \
                               --denticity monodentate \
                               --train_mode None \
                               --labeling isolated_ligands
