# Calculator on Rasberry Pi using PiCam & handwritten tasks

This is a project being done in the "Deep Learning on Raberry Pi" class, offered by ETH Zurich and UZH. It's a calculator, installed on Rasberry Pi and using state of the art Deep Learning and Computer Vision techniques to recognize handwritten calculations, computing them and finally output the result through a built-in speaker.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them

```
numpy
tensorflow
pnslib
python 3
```

### Installing

To install the requireded libraries just follow the instructions on the website to the PnS website
* [PnS Website](https://pns2019.github.io/python-sc.html)

There is a step by step tutorial on how to install numpy, tensorflow, python 3 and the pnslib on the desired machine both for Linux and Mac.


## Deployment
Download the trainedmodel.hdf5 file to your machine

If you want to train the model by yourself download the dataset that we used here:
* [datatset](https://www.kaggle.com/xainano/handwrittenmathsymbols)

After that create individual folders on your machine with the desired digits and signs to train the model and change the datatpaths in modeltraining.py to your specific terms
Use the terminal and run modeltraining.py
```
cd "to the modeltraining.py file"
python3 modeltraining.py
```
change the datapaths in main.py fitting to the machine you're operating on
Use terminal go to the location of the main.py file
```
cd "to the main.py file"
python3 main.py
```

## Built With

* [Tensorflow](https://www.tensorflow.org) - Deep Learning ecosystem used
* [Keras](https://keras.io) - API
* [Python](https://www.python.org) - The almighty language


## Authors

* **Roman Flepp** - *Computer Vision*
* **Emil Funke** - *Data Management*
* **Luc von Niederhäusern** - *Backend*

## License

This project is licensed under the ETH License

## Acknowledgments

* Big thank you to our TAs Yuhuang Hu and Iulia-Alexandra Lungu for helping us out with basically everything
* Thanks to Xai Nano for his mathematical symbols dataset

