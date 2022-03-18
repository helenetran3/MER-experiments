# Updated version of [Multimodal DNN](https://github.com/rhoposit/MultimodalDNN)

:dart: This repository is intended to update the code of [Multimodal DNN](https://github.com/rhoposit/MultimodalDNN) and make 
it run with Python 3 and the current version of [CMU MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK). 
This README will be updated once the code works. Thank you for your interest!

[//]: # (Paper: [Recognizing Emotions in Video Using Multimodal DNN Feature Fusion]&#40;http://www.aclweb.org/anthology/W18-3302&#41;)

Please cite the paper of the original authors if the code was useful to you:
```
@inproceedings{williams2018a,
  title     = "Recognizing Emotions in Video Using Multimodal DNN Feature Fusion",
  author    = "Jennifer Williams and Steven Kleinegesse and Ramona Comanescu and Oana Radu",
  year      = "2018",
  pages     = "11--19",
  booktitle = "Proceedings of Grand Challenge and Workshop on Human Multimodal Language (Challenge-HML)",
  publisher = "Association for Computational Linguistics",
}
```

Note that we only focus on the code for **multimodal emotion recognition** using CMU-MOSEI dataset. The handcrafted 
features provided by the SDK are **OpenFace 2** (image), **FACET 4.2** (image), **COVAREP** (audio), 
**glove_vectors** (text). Some slight changes have been made from the original repository 
(cf. [Differences with the original code](#Differences-with-the-original-code) section).

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
    usage: main.py [-h] [-df DATASET_FOLDER] [-pn PICKLE_NAME] [-pf PICKLE_FOLDER] [-t] [-al]
                   [-c] [-v {loss,acc}] [-f {facet,openface}] [-b BATCH_SIZE]
                   [-s FIXED_NUM_STEPS] [-l {1,2,3}] [-n NUM_NODES] [-d DROPOUT_RATE]
                   [-a FINAL_ACTIV]
    ```
    
    Optional arguments:
    
    ```commandline
      -h, --help            show this help message and exit
      -df DATASET_FOLDER, --dataset_folder DATASET_FOLDER
                            Name of the folder where the CMU-MOSEI mmdataset will be
                            downloaded.
      -pn PICKLE_NAME, --pickle_name PICKLE_NAME
                            Name of the pickle object that will contain the CMU-MOSEI
                            mmdataset.
      -pf PICKLE_FOLDER, --pickle_folder PICKLE_FOLDER
                            Name of the folder where to save the pickle object that contain
                            the CMU-MOSEI mmdataset.
      -t, --align_to_text   Data will be aligned to the textual modality.
      -al, --append_label_to_data
                            Append annotations to the dataset.
      -c, --with_custom_split
                            Perform custom split on training and validation sets (for more
                            details, cf. paper).
      -v {loss,acc}, --val_metric {loss,acc}
                            Metric to monitor for validation set. Values: loss or acc.
      -f {facet,openface}, --image_feature {facet,openface}
                            Image features. Values: facet or openface.
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            Batch size
      -s FIXED_NUM_STEPS, --fixed_num_steps FIXED_NUM_STEPS
                            Number of steps to fix for all sequences. Set to 0 if you want to
                            keep the original number of steps.
      -l {1,2,3}, --num_layers {1,2,3}
                            Number of bidirectional layers. Values between 1 and 3.
      -n NUM_NODES, --num_nodes NUM_NODES
                            Number of nodes in the penultimate dense layer.
      -d DROPOUT_RATE, --dropout_rate DROPOUT_RATE
                            Dropout rate
      -a FINAL_ACTIV, --final_activ FINAL_ACTIV
                            Activation function of the final layer.

    ```


## Differences with the original code


|            | Original Code                     | Current Code                                                   |
|------------|-----------------------------------|----------------------------------------------------------------|
| **Inputs** | Only training and validation sets | Standard training, validation and test sets from CMU-MOSEI SDK |


