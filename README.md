# CNN-hybrid-models
This is an on-going project which explores CNN hybrid architectures. The customized architectures will be compared with standard models such as resNet50, MobileNetV2, Densenet121 and more. This repo will host the code, observations and the paper that discusses the findings. 
<pre>
leafnet_project/
│── data/
│    ├── new_plant_diseases/   # training + validation dataset
│    └── plantvillage/         # test dataset only
│
│── models/
│    ├── leafnetv2.py          
│    ├── mobilenet.py          
│    ├── efficientnet.py       
│
│── utils/
│    ├── datasets.py           # loaders, transforms, augmentations
│    ├── metrics.py            # accuracy, macro-F1, confusion matrix
│    ├── train_utils.py        # train/validate loops
│
│── train.py                   
│── test.py                    
│── ablation.py                
│── requirements.txt
<pre>

For testing:
python test.py --model leafnet
python test.py --model efficientnetv2
python test.py --model mobilenet

For Training:
python train.py --model leafnet
python train.py --model efficientnetv2
python train.py --model mobilenet
