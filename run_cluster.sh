#!/bin/bash 

#===============================================================================
# exemples d'options
 
#SBATCH --partition=court     # choix de la partition où soumettre le job
#SBATCH --ntasks=1            # nb de tasks total pour le job
#SBATCH --cpus-per-task=20    # nb CPU pour une task
#SBATCH --mem=64000           # mémoire nécessaire (par noeud) en Mo
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=helene.tran@doctorant.uca.fr
 
#===============================================================================

python3 main.py \
--pickle_name_dataset cmu_mosei_aligned_text_no_averaged \
--pickle_name_fold fold \
--align_to_text \
--append_label_to_data \
--val_metric loss \
--image_feature facet \
--batch_size 32 \
--fixed_num_steps 45 \
--num_layers 1 \
--num_nodes 100 \
--dropout_rate 0.2 \
--final_activ linear \
--model_name MultimodalDNN \
--num_epochs 2 \
--patience 1 \
--learning_rate 0.001 \
--loss_function mean_absolute_error \
--threshold_emo_present 0 \
--round_decimals 3 \
--predict_neutral_class
#--with_custom_split \
