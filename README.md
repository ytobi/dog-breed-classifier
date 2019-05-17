# Dog Breed Classifier

In this project, I built a pipeline to process real-world, user-supplied images. Given an image of a dog, our algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, our code will identify the resembling dog breed. The project is part of fulfillment for a nanodegree at Udacity.

# Installation

For best experience with managing dependency I advise you install [Anconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/download.html).

Create a virtual environment with conda
```
conda create --name deep-learning
```
Activate environment.
```
source activate deep-learning
```

Install dependencies.

```
pip install -r requirements.txt
```

Download or clone this Dog_Breed_Classifier repository. Launch the app with jupyter-notebook.
```
jupyter-notebook dog_app.py
```

# Usage

### Setup
Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and place it in this project's home directory, at the location /dog_images.

Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and place it in the home directory, at location /lfw

Note: If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.

### Run
Run all code cells in the notebook (This will take a very very long time to run on a CPU as the code will construct and train a deep convolutional neural network, preferably(for Goodness sake) you should run on a GPU).
Test the app by passing a file path of your own image(s) to the `run_app` function. The `run_app` function predicts the breed of dog for the image provided if a human or any other image is provided the model predicts the resembling dog breed.


The trained model achieved 81% accuracy in prediction canine breed.




