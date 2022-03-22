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

python3 ./main.py -pickle_name cmu_mosei_aligned_text_no_averaged 
