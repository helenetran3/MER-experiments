# Updated version of [Multimodal DNN](https://github.com/rhoposit/MultimodalDNN)

:dart: This repository is intended to update the code of [Multimodal DNN](https://github.com/rhoposit/MultimodalDNN) and make 
it run with Python 3 and the current version of [CMU MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK). 
This README will be updated once the code works. Thank you for your interest!

Paper: [Recognizing Emotions in Video Using Multimodal DNN Feature Fusion](http://www.aclweb.org/anthology/W18-3302)

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

## How to run the code
1. Clone [CMU MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) and follow the installation steps outlined there.
2. Run as follows:
```commandline
python3 main.py [-FLAGS]
```

#### List of the flags

| Flag name            | Values            | Description                                                                                                                       | Default                 |
|----------------------|-------------------|-----------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| dataset_folder       | str               | Name of the folder where the CMU-MOSEI mmdataset will be downloaded                                                               | cmu_mosei/              |
| pickle_name          | str               | Name of the pickle object that will contain the CMU-MOSEI mmdataset                                                               | cmu_mosei               |
| pickle_folder        | str               | Name of the folder where to save the pickle object that contain the CMU-MOSEI mmdataset                                           | cmu_mosei/pickle_files/ |
| align_to_text        | {0, 1}            | Whether we want data to align to the textual modality. 1 for True and 0 for False                                                 | 1 (True)                |
| append_label_to_data | {0, 1}            | Whether we want data to append annotations to the dataset. 1 for True and 0 for False                                             | 1 (True)                |
| with_custom_split    | {0, 1}            | Whether we want to perform custom split on training and validation sets (for more details, cf. paper). 1 for True and 0 for False | 0 (False)               |
| val_metric           | {loss, acc}       | Metric to monitor for validation set                                                                                              | loss                    |
| image_feature        | {facet, openface} | Image features: FACET 4.2 or OpenFace 2                                                                                           | facet                   |



## Differences with the original code


|            | Original Code                     | Current Code                                                   |
|------------|-----------------------------------|----------------------------------------------------------------|
| **Inputs** | Only training and validation sets | Standard training, validation and test sets from CMU-MOSEI SDK |


