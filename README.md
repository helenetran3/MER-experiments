# Multimodal Emotion Recognition SOTA models

:dart: This repository is intended to experiment state-of-the-art codes on Multimodal Emotion Recognition.

Feel free to open an issue on this repository if you have any question or remark. Thank you for your interest!


### README Structure

1. [CMU-MOSEI Database](#CMU-MOSEI-Database)
2. [Environment](#Environment)
3. [How to run the code](#How-to-run-the-code)

## CMU-MOSEI Database

The CMU - *Multimodal Opinion Sentiment and Emotion Intensity* database is one of the key databases in Multimodal Emotion
Recognition.

  <p font="italic" align="center">  

  <img src="images/cmu_mosei.png" align="center" alt="cmu_mosei_database">

  </p>

  <p font="italic" align="center">
  Source: http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/
  </p>

- Modalities: Audio, Video, Text (transcript)
- Emotions: Anger, Disgust, Fear, Happiness, Sadness, Surprise + Neutral
- 23 453 YouTube videos covering 250 topics
- 1000 speakers facing the camera
- Speech language: English
- Wild emotions

The authors provide a SDK that can be found here: [CMU-Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).
The handcrafted features provided by the SDK are **OpenFace 2** (image), **FACET 4.2** (image), **COVAREP** (audio), 
**glove_vectors** (text).

```commandline
@inproceedings{Zadeh2018,
  title={{Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph}},
  author={Zadeh, AmirAli Bagher and Liang, Paul Pu and Vanbriesen, Jonathan and Poria, Soujanya and Tong, Edmund and Cambria, Erik and Chen, Minghai and Morency, Louis-Philippe},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  pages={2236--2246},
  year={2018}
}
```

## Environment
- Python 3.9.7
- TensorFlow 2.8
- Keras 2.8
- NumPy 1.20.3


## How to run the code
1. Clone [CMU MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) and follow the installation steps outlined there.
2. Two ways to run the code:

   1. **Using bash script** (you can set the parameters by editing the file):
   
    ```commandline
    ./run_cluster.sh
    ```
    

   2. **Running main.py in the command line:**

    ```commandline
      usage: main.py [-h] [-df DATASET_FOLDER] [-pnd PICKLE_NAME_DATASET] [-pnf PICKLE_NAME_FOLD]
               [-pf PICKLE_FOLDER] [-t] [-al] [-c] [-v {loss,acc}] [-f {facet,openface}]
               [-b BATCH_SIZE] [-s FIXED_NUM_STEPS] [-l {1,2,3}] [-n NUM_NODES] [-d DROPOUT_RATE]
               [-a FINAL_ACTIV] [-mf MODEL_FOLDER] [-mn MODEL_NAME] [-cf CSV_FOLDER] [-cn CSV_NAME]
               [-e NUM_EPOCHS] [-p PATIENCE] [-lr LEARNING_RATE] [-lf LOSS_FUNCTION]
               [-rd ROUND_DECIMALS]

      SOTA Multimodal Emotion Recognition models using CMU-MOSEI database.
      
      optional arguments:
        -h, --help            show this help message and exit
        -df DATASET_FOLDER, --dataset_folder DATASET_FOLDER
                              Name of the folder where the CMU-MOSEI mmdataset will be downloaded.
        -pnd PICKLE_NAME_DATASET, --pickle_name_dataset PICKLE_NAME_DATASET
                              Name of the pickle object that will contain the CMU-MOSEI mmdataset.
        -pnf PICKLE_NAME_FOLD, --pickle_name_fold PICKLE_NAME_FOLD
                              Name of the pickle object that will contain the training, validation and test
                              folds.
        -pf PICKLE_FOLDER, --pickle_folder PICKLE_FOLDER
                              Name of the folder where to save the pickle object that contain the CMU-MOSEI
                              mmdataset.
        -t, --align_to_text   Data will be aligned to the textual modality.
        -al, --append_label_to_data
                              Append annotations to the dataset.
        -c, --with_custom_split
                              Perform custom split on training and validation sets (for more details, cf.
                              paper).
        -v {loss,acc}, --val_metric {loss,acc}
                              Metric to monitor for validation set. Values: loss or acc.
        -f {facet,openface}, --image_feature {facet,openface}
                              Image features. Values: facet or openface.
        -b BATCH_SIZE, --batch_size BATCH_SIZE
                              Batch size
        -s FIXED_NUM_STEPS, --fixed_num_steps FIXED_NUM_STEPS
                              Number of steps to fix for all sequences. Set to 0 if you want to keep the
                              original number of steps.
        -l {1,2,3}, --num_layers {1,2,3}
                              Number of bidirectional layers. Values between 1 and 3.
        -n NUM_NODES, --num_nodes NUM_NODES
                              Number of nodes in the penultimate dense layer.
        -d DROPOUT_RATE, --dropout_rate DROPOUT_RATE
                              Dropout rate
        -a FINAL_ACTIV, --final_activ FINAL_ACTIV
                              Activation function of the final layer.
        -mf MODEL_FOLDER, --model_folder MODEL_FOLDER
                              Name of the directory where the models will be saved.
        -mn MODEL_NAME, --model_name MODEL_NAME
                              Name of the model to be saved.
        -cf CSV_FOLDER, --csv_folder CSV_FOLDER
                              Name of the directory where the csv file containing the results is saved.
        -cn CSV_NAME, --csv_name CSV_NAME
                              Name of the csv file.
        -e NUM_EPOCHS, --num_epochs NUM_EPOCHS
                              Maximum number of epochs
        -p PATIENCE, --patience PATIENCE
                              Number of epochs with no improvement after which the training will be stopped.
        -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                              Learning rate
        -lf LOSS_FUNCTION, --loss_function LOSS_FUNCTION
                              Loss function
        -rd ROUND_DECIMALS, --round_decimals ROUND_DECIMALS
                              Number of decimals to be rounded for metrics.

    ```

## Models available

*The current code runs only the following model for the moment.*

- [Recognizing Emotions in Video Using Multimodal DNN Feature Fusion](http://www.aclweb.org/anthology/W18-3302) 
(updated version of [Multimodal DNN](https://github.com/rhoposit/MultimodalDNN))
   - Use standard training, validation and test sets from CMU-MOSEI SDK (instead of using only training and validation sets 
originally specified in the paper)


