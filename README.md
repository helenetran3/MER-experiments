**Note from helenetran3:** This forked repository is intended to update the code and make it run. This README will be updated once the code works using the current version of [CMU MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK). 

Steps:
1. Clone [CMU MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) and follow the installation steps outlined there.
2. Run the code using **Python 2.7** as specified below (I personally use Conda to create an environment with the compatible librairies).

*The following text is written by the original authors.*

# MultimodalDNN

This repository provides code from the University of Edinburgh Team G25 for the ACL 2018 [Workshop on Human Multimodal Language](http://multicomp.cs.cmu.edu/acl2018multimodalchallenge/).

### EMOTION - First Place in Emotion Recognition Challenge (all metrics) using MOSEI data

Paper: [Recognizing Emotions in Video Using Multimodal DNN Feature Fusion](http://www.aclweb.org/anthology/W18-3302)

Code: emotion_recognition.py

Run: `emotion_recognition.py [mode]`

Where `[mode]` specifies the multimodal inputs (A=Audio, V=Video, T=Text): `all`, `AV`, `AT`, `VT`, `V`, `T`, or `A`

This script will run a sweep of all parameters described in our paper, including number of BLSTM layers and dropout rates. It is designed to run the sweep in parallel and thus requires a significant compute resource. 

To cite (BibTeX):
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

### SENTIMENT - Multimodal Sentiment Analysis using MOSI data

Paper: [DNN Multimodal Fusion Techniques for Predicting Video Sentiment](http://www.aclweb.org/anthology/W18-3309)

Code: MOSI_*.py

Run: `MOSI_*.py [mode] [task]`

Where `[mode]` specifies the multimodal inputs (A=Audio, V=Video, T=Text): `all`, `AV`, `AT`, `VT`, `V`, `T`, or `A`
and `[task]` specifies if the task is binary, 5-class, or regression.

This script will run a sweep of all parameters described in our paper.


To cite (BibTeX):
```
@inproceedings{williams2018b,
  title     = "DNN Multimodal Fusion Techniques for Predicting Video Sentiment",
  author    = "Jennifer Williams and Ramona Comanescu and Oana Radu and Leimin Tian",
  year      = "2018",
  pages     = "64--72",
  booktitle = "Proceedings of Grand Challenge and Workshop on Human Multimodal Language (Challenge-HML)",
  publisher = "Association for Computational Linguistics",
}
```


### Notes:
1. Our code is designed to interface with the [CMU MultiModalDataSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK). Do cite their dataset along with our paper.
2. To work with the MOSEI dataset in particular, the dataset is currently very large, and you require a large amount of RAM 
3. If you have questions about this code, please open an issue on this repository. 
4. If you have questions related to the data itself, please contact the CMU team.
5. This code is provided as-is, and is the code used for our University of Edinburgh Team G25 submission.
