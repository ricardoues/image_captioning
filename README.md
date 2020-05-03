# Image Captioning Project of Computer Vision Nanodegree from Udacity

## Introduction
This project is about to implement a image captioning using Convolutional Neural Networks and Recurrent Neural Networks. Our approach is to use the inject model which is explained in the following web page: 

[Caption Generation with the Inject and Merge Encoder-Decoder Models ](https://machinelearningmastery.com/caption-generation-inject-merge-architectures-encoder-decoder-model/)

## How to prepare the Amazon AMI in order to run the models 

Use the following AMI from AWS Market Place: 

[AISE PyTorch 0.4 Python 3.6 CUDA 9.1 Notebook](https://aws.amazon.com/marketplace/pp/Jetware-AISE-PyTorch-04-Python-36-CUDA-91-Notebook/B07D2J9V8V#pdp-usage)

After that, clone the repository:

```bash
git clone https://github.com/ricardoues/image_captioning.git
```

Run the requeriments file as follows:


```bash
pip3 install -r requirements.txt
```
Install the COCO library according this webpage:

[Install Coco Library](https://github.com/udacity/CVND---Image-Captioning-Project)

Note: In the step 2 you must run the following commands:


```bash
cd cocoapi/PythonAPI  
make  
python3 setup.py build
python3 setup.py install
```

The folder cocoapi must be in /opt.  
