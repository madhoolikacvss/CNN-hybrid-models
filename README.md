# CNN-hybrid-models
This is an on-going project which explores CNN hybrid architectures. The customized architectures will be compared with standard models such as resNet50, MobileNetV2, Densenet121 and more. This repo will host the code, observations and the paper that discusses the findings. 
<pre>
leafnet_project/
│── data/
│    ├── new_plant_diseases/   # training + validation dataset
│    └── plantvillage/         # test dataset only
│
│── models/
│    ├── leafnetv2.py          # your hybrid CNN
│    ├── mobilenet.py          # baseline wrapper
│    ├── efficientnet.py       # baseline wrapper
│
│── utils/
│    ├── datasets.py           # loaders, transforms, augmentations
│    ├── metrics.py            # accuracy, macro-F1, confusion matrix
│    ├── train_utils.py        # train/validate loops
│
│── train.py                   # main script for training & validation
│── test.py                    # run evaluation on PlantVillage
│── ablation.py                # run ablation studies
│── requirements.txt
<pre>
