## ULTIMA-OCT

This repository contains the code used for reconstruction in the presented paper 
together with the down-sampled dataset used for Fig. 5. 

## System requirements

The software is written in Python and requires the packages jax and equinox to run. 
It is designed to run on a graphics processing unit (GPU).

## Installation guide

We recommend using Google Colab to run the code, since it comes with GPU acceleration 
and most of the requirements already installed. The runs with the free version of 
Google Colab.

## Instruction for the demo

We included a jupyter notebook that runs on the down-sampled data from Fig. 5. 
and performs the reconstruction of reflectivity, attenuation and refractive index 
contrast as well as alignment of the data. The notebook can be found at

https://colab.research.google.com/drive/1YiXc7GJ4jXwCeoLLxgQyqhUcQm1T8n2x?usp=sharing

First unpack the contents of the zip file into your Google Drive. 
When running the code on google colab, under "Runtime" select "Change runtime type", then select "T4 GPU". 
This makes it possible to leverage GPU acceleration. With these settings, the notebook provided should take 
approximately 25 minutes to run on google colab on a runtime with a T4 GPU enabled.