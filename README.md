# One Class Classification  
 Implementation of one class classifier using Keras, to detect bananas and everything else  


## General approach:  
- Collect data (Kaggle)  
    - The dataset I use is made of 500x bananas and also a dataset of 1500x randoms images since it's needed to classify bananas and others things  


- Pre-processing (OpenCV, Keras)  
    - Resize images to 224*224 (height and width of the data used in the pre-trained model)  
    - Remove transparency from images  
    - Converts RGB to grayscale, but still keep 3 dimensions for later use with MNet transfer learning  
    - Data augmentation, create news images from that I have with performing alteration (flip, rotation, translation, contrast)  


- Deep learning (Keras)
    - Train a custom CNet model  
    - Use a pre-trained model to extract features therefor train a DNN  
    - Compare results  


- Create an interface to deploy model (Dash)  


- Deploy the interface to be accessible online (Heroku)  
The interface is accessible from this [link](https://occ-bananas.herokuapp.com/) (it might takes several seconds to load the page if it has not been loaded since a while)  

## Content

- `/jupyter`: 2 notebooks used to [pre-process](jupyter/preprocess.ipynb) the data and [train models](jupyter/models.ipynb)  
- `/python`: Dash interface that uses requirements.txt  
- `/asset`: essentially 3 different Keras trained models  

### Remarks
The dataset I used is really biased, most of the bananas image have a white background so the training will lead to learn that is common for a banana image to have many white pixel.  
A biased model that learned from this data will perform good excellent on train, test, validation but very poor results on general predictions.  
That's why I decided to apply a grayscale to each image and try to reduce this background bias.  


## Improvements
Display an overview of some recent predictions on the interface through an external database :banana:  
Collect more data and remake the training :heavy_multiplication_x:  
Multiple bananas predictions from the interface :heavy_multiplication_x:  
Collect uploaded data from prediction on the interface for later use (re-train models) :heavy_multiplication_x:
Display pre-processing step by step on the interface :heavy_multiplication_x:  
Give user ability to supervise the prediction and save his supervision (basically correct/incorrect) :heavy_multiplication_x:


### *References*:
https://occ-bananas.herokuapp.com/  
https://www.tensorflow.org/tutorials/images/data_augmentation  
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html    
https://keras.io/api/applications/  
