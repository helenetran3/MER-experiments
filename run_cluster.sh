#!/bin/bash 

#===============================================================================
# exemples d'options
 
#SBATCH --partition=court     # choix de la partition où soumettre le job
#SBATCH --time=48:0:0         # temps max alloué au job (format = m:s ou h:m:s ou j-h:m:s)
#SBATCH --ntasks=1            # nb de tasks total pour le job
#SBATCH --cpus-per-task=24    # nb CPU pour une task
#SBATCH --mem=1000            # mémoire nécessaire (par noeud) en Mo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=helene.tran@doctorant.uca.fr
 
#===============================================================================

python3 main.py \
--dataset_folder cmu_mosei/ \
--pickle_name_dataset cmu_mosei \
--pickle_name_fold fold \
--pickle_folder cmu_mosei/pickle_files/ \
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
--model_folder models/ \
--model_name MultimodalDNN \
--num_epochs 200 \
--patience 1 \
--learning_rate 0.001 \
--loss_function mean_absolute_error
#--with_custom_split \
