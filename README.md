# AISNA

## Description 

**AISNA** (or *Admin IS Not Admin*) is a project inspired from the Person of Interest TV series and more specifically from this excerpt from an episode (S4xE05):  

[![Admin is not Admin](http://img.youtube.com/vi/nhWe2nf24ag/0.jpg)](http://www.youtube.com/watch?v=nhWe2nf24ag "Person of Interest - Admin is not Admin")

AISNA is a deep learning algorithm which use **live faces recognition** to determine if the person on camera is either an admin or an unknown person. It also uses **voice recognition** to show more features on the broadcast if certain words are used.

## Deep Learning model

### Model

This project uses MobileNet model to identify the face of the person in front of the webcam.

MobileNets are a family of mobile-first computer vision models for TensorFlow, designed to effectively maximize accuracy.  
They are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases.

It was easier and faster to use MobileNets compared to other models such as ResNet50, ResNet152 or Xception who has longer execution time for similar results.

#### Architecture of MobileNets

![MobileNets architecture](https://miro.medium.com/max/570/1*TJAjuueT9_pk2Nlv1zmb4A.png)

### Fine-tuning parameters

I chose to fine-tune this different parameters on the MobileNet architecture: 

- Batch Size (values: 16, 64, 128, 256)
- Activation function (values: Sigmoid, ReLu, Softmax)
- Optimizer (values: Adam, RMSProp, SGD)
- Learning Rate (values: 0.001, 0.01, 0.05, 0.1, 0.5)

### Results

## Getting Started

### Prerequisites

The virtualenv package is required to create virtual environments. You can install it with pip:
```sh
    pip install virtualenv
```

### Installation

1. Clone the repo
```sh
    git clone https://github.com/Scylidose/AISNA.git
```

2. Create virtual environment
```sh
    virtualenv myvenv
```

3. Activate virtual environment

```sh
    .\myvenv\\Scripts\activate
```

4. Install requirements

```sh
    pip install -r requirements.txt
```

## References

https://towardsdatascience.com/how-to-create-real-time-face-detector-ff0e1f81925f

https://realpython.com/face-recognition-with-python/

https://realpython.com/face-detection-in-python-using-a-webcam/

https://face-recognition.readthedocs.io/en/latest/readme.html 

https://www.mygreatlearning.com/blog/face-recognition/

https://realpython.com/python-speech-recognition/

https://towardsdatascience.com/real-time-speech-recognition-python-assemblyai-13d35eeed226

https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470