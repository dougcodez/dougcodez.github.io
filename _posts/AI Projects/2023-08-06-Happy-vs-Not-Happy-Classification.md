---
tag: AI Projects
title: Happy vs Not Happy Classification
---

# Happy vs Not Happy Classification
# About
This project focuses on the classification of a set of images into two categories: happy and not happy. By employing convolutional models, we will be able to effectively classify these images and gain insights into the distinguishing features that contribute to their respective labels.

# Dataset Information
Below represents the information about the dataset we worked with:
| Category | Dataset Category | Image Amount | 
|-----------|------------------|--------------|
| Happy     | Training         |  14          |
| Not Happy | Training         |  14          |
| Happy     | Validation       |  5           |
| Not Happy | Validation       |  5           |
| Mix       | Testing          |  3           |

Now that you have an idea of what the dataset looks like lets get into the code!

## Introduction 
First we need to unzip our given zip folder that withholds all our pictures, and expect a given output like so:
``` python
!unzip '/content/happy vs not happy.zip'
```
![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/02ff4230-232b-42bd-a821-5e1626f3b8b7)

With the images extracted, the next step is to import the libraries and modules needed for our data analysis and machine learning tasks. Here are the given imports we 
need for this project: 

``` python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
```

Now that we have our extracted images and imports we are good to move on to the next step!

## Data Preparation 
In this phase, we'll prepare the image data for training and validation. The process involves loading images, resizing them, and creating data generators that will feed data
to our model. First we'll demonstrate loading and displaying an image using the load_img that comes from the Keras library. The image being plotted comes from our training/happy directory.
``` python
img = image.load_img('/content/happy vs not happy/training/happy/3.png')
plt.imshow(img)
plt.show()
```
![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/3e38bc41-0f16-492f-95a9-0b1bd02e8d5b)

To get an idea what the shape of our image is we use the cv2.imread() function to get an idea of what it is 
``` python
cv2.imread('/content/happy vs not happy/training/happy/3.png').shape
```
![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/f8456215-26b7-4d56-8a2b-bd5c95e7881e)

Moving on, we'll create data generators using the ImageDataGenerator class from Keras. These generators will preprocess and augment our image data on-the-fly during training and validation
```python
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)
```
We'll then use the data generators to create training and validation datasets. The flow_from_directory function automatically organizes and labels our images based on their folder structure.
This is useful when dealing with image classification tasks. This is what it would look like in code for our training dataset and validation dataset, printing the given output:
``` python
train_ds = train.flow_from_directory('/content/happy vs not happy/training/',
                                  target_size=(250,250),
                                  batch_size=3,
                                  class_mode='binary')

val_ds = validation.flow_from_directory('/content/happy vs not happy/validation/',
                                  target_size=(250,250),
                                  batch_size=3,
                                  class_mode='binary')
```
![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/dfc78c87-45a3-4c76-8544-7041e34ecf87)

Here we get an idea on what our class indicies and classes are.
``` python
train_ds.class_indices
```
![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/c49c8dfd-2336-47ee-89d2-8ee779d2b66e)

``` python
train_ds.classes
```
![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/693180d7-5752-4ab5-a1aa-f8626ff91150)

## Model Training
With our data preprocessed and organized, we can now move on to building and training our maching learning model. 
```python
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(250, 250, 3)),
    MaxPool2D(2, 2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPool2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer= Adam(lr=0.001), metrics=['accuracy'])
model.fit(train_ds, steps_per_epoch=3, epochs=100, validation_data=val_ds)
```
We define a sequential mode using Keras Sequential class. The model consists of a series of convolutional and pooling layers followed by fully connected layers. 
The given input shape is 250x250x3 so that the full image is fully captured when the model is training. This architecture captures hierachial features from the input
images. We then compile and fit our model with the appropriate parameters. The results are as given:

``` python
[ ]
model.fit(train_ds, steps_per_epoch=3, epochs=100, validation_data=val_ds)
Epoch 1/100
3/3 [==============================] - 4s 267ms/step - loss: 6.7676 - accuracy: 0.4444 - val_loss: 4.3946 - val_accuracy: 0.5000
Epoch 2/100
3/3 [==============================] - 0s 71ms/step - loss: 3.3757 - accuracy: 0.4444 - val_loss: 0.8412 - val_accuracy: 0.5000
Epoch 3/100
3/3 [==============================] - 0s 103ms/step - loss: 0.8702 - accuracy: 0.2222 - val_loss: 0.6660 - val_accuracy: 0.5000
Epoch 4/100
3/3 [==============================] - 0s 101ms/step - loss: 0.6294 - accuracy: 0.7778 - val_loss: 0.7074 - val_accuracy: 0.5000
Epoch 5/100
3/3 [==============================] - 0s 97ms/step - loss: 0.7173 - accuracy: 0.4444 - val_loss: 0.6674 - val_accuracy: 0.6000
Epoch 6/100
3/3 [==============================] - 0s 68ms/step - loss: 0.6309 - accuracy: 0.8889 - val_loss: 0.6440 - val_accuracy: 0.5000
Epoch 7/100
3/3 [==============================] - 0s 67ms/step - loss: 0.5903 - accuracy: 0.5556 - val_loss: 0.6122 - val_accuracy: 0.6000
Epoch 8/100
3/3 [==============================] - 0s 67ms/step - loss: 0.4183 - accuracy: 1.0000 - val_loss: 0.7047 - val_accuracy: 0.5000
Epoch 9/100
3/3 [==============================] - 0s 97ms/step - loss: 0.6933 - accuracy: 0.5714 - val_loss: 0.7157 - val_accuracy: 0.5000
Epoch 10/100
3/3 [==============================] - 0s 102ms/step - loss: 0.5487 - accuracy: 0.7778 - val_loss: 0.6069 - val_accuracy: 0.6000
Epoch 11/100
3/3 [==============================] - 0s 67ms/step - loss: 0.3650 - accuracy: 0.7778 - val_loss: 0.5568 - val_accuracy: 0.6000
Epoch 12/100
3/3 [==============================] - 0s 66ms/step - loss: 0.3772 - accuracy: 0.8889 - val_loss: 0.5099 - val_accuracy: 0.7000
Epoch 13/100
3/3 [==============================] - 0s 64ms/step - loss: 0.3396 - accuracy: 0.8889 - val_loss: 0.4617 - val_accuracy: 0.8000
Epoch 14/100
3/3 [==============================] - 0s 64ms/step - loss: 0.4068 - accuracy: 0.8889 - val_loss: 0.5580 - val_accuracy: 0.7000
Epoch 15/100
3/3 [==============================] - 0s 67ms/step - loss: 0.6330 - accuracy: 0.5714 - val_loss: 0.3445 - val_accuracy: 0.9000
Epoch 16/100
3/3 [==============================] - 0s 64ms/step - loss: 0.4687 - accuracy: 0.5556 - val_loss: 0.5390 - val_accuracy: 0.7000
Epoch 17/100
3/3 [==============================] - 0s 57ms/step - loss: 0.2621 - accuracy: 0.7778 - val_loss: 0.8454 - val_accuracy: 0.6000
Epoch 18/100
3/3 [==============================] - 0s 97ms/step - loss: 0.3878 - accuracy: 0.7778 - val_loss: 0.2364 - val_accuracy: 1.0000
Epoch 19/100
3/3 [==============================] - 0s 67ms/step - loss: 0.2686 - accuracy: 0.8571 - val_loss: 0.1866 - val_accuracy: 1.0000
Epoch 20/100
3/3 [==============================] - 0s 60ms/step - loss: 0.3115 - accuracy: 0.8889 - val_loss: 0.1463 - val_accuracy: 1.0000
Epoch 21/100
3/3 [==============================] - 0s 98ms/step - loss: 0.1392 - accuracy: 1.0000 - val_loss: 0.1201 - val_accuracy: 1.0000
Epoch 22/100
3/3 [==============================] - 0s 61ms/step - loss: 0.0779 - accuracy: 1.0000 - val_loss: 0.0649 - val_accuracy: 1.0000
Epoch 23/100
3/3 [==============================] - 0s 56ms/step - loss: 0.0555 - accuracy: 1.0000 - val_loss: 0.0761 - val_accuracy: 1.0000
Epoch 24/100
3/3 [==============================] - 0s 61ms/step - loss: 0.0220 - accuracy: 1.0000 - val_loss: 1.9045 - val_accuracy: 0.5000
Epoch 25/100
3/3 [==============================] - 0s 75ms/step - loss: 0.1753 - accuracy: 0.8889 - val_loss: 0.0290 - val_accuracy: 1.0000
Epoch 26/100
3/3 [==============================] - 0s 66ms/step - loss: 0.5908 - accuracy: 0.7143 - val_loss: 0.0476 - val_accuracy: 1.0000
Epoch 27/100
3/3 [==============================] - 0s 107ms/step - loss: 0.0875 - accuracy: 1.0000 - val_loss: 0.1221 - val_accuracy: 1.0000
Epoch 28/100
3/3 [==============================] - 0s 99ms/step - loss: 0.2083 - accuracy: 0.8889 - val_loss: 0.1971 - val_accuracy: 1.0000
Epoch 29/100
3/3 [==============================] - 0s 58ms/step - loss: 0.1657 - accuracy: 1.0000 - val_loss: 0.2133 - val_accuracy: 1.0000
Epoch 30/100
3/3 [==============================] - 0s 63ms/step - loss: 0.1864 - accuracy: 1.0000 - val_loss: 0.1072 - val_accuracy: 1.0000
Epoch 31/100
3/3 [==============================] - 0s 62ms/step - loss: 0.1199 - accuracy: 1.0000 - val_loss: 0.0646 - val_accuracy: 1.0000
Epoch 32/100
3/3 [==============================] - 0s 62ms/step - loss: 0.0428 - accuracy: 1.0000 - val_loss: 0.0510 - val_accuracy: 1.0000
Epoch 33/100
3/3 [==============================] - 0s 68ms/step - loss: 0.0154 - accuracy: 1.0000 - val_loss: 0.0255 - val_accuracy: 1.0000
Epoch 34/100
3/3 [==============================] - 0s 107ms/step - loss: 0.1020 - accuracy: 0.8889 - val_loss: 0.2716 - val_accuracy: 0.9000
Epoch 35/100
3/3 [==============================] - 0s 107ms/step - loss: 0.3216 - accuracy: 0.8889 - val_loss: 0.0177 - val_accuracy: 1.0000
Epoch 36/100
3/3 [==============================] - 0s 95ms/step - loss: 0.0030 - accuracy: 1.0000 - val_loss: 0.3155 - val_accuracy: 0.8000
Epoch 37/100
3/3 [==============================] - 0s 69ms/step - loss: 0.4413 - accuracy: 0.7778 - val_loss: 0.0474 - val_accuracy: 1.0000
Epoch 38/100
3/3 [==============================] - 0s 62ms/step - loss: 0.0506 - accuracy: 1.0000 - val_loss: 0.4350 - val_accuracy: 0.7000
Epoch 39/100
3/3 [==============================] - 0s 65ms/step - loss: 0.2488 - accuracy: 0.8889 - val_loss: 0.0667 - val_accuracy: 1.0000
Epoch 40/100
3/3 [==============================] - 0s 108ms/step - loss: 0.0363 - accuracy: 1.0000 - val_loss: 0.0380 - val_accuracy: 1.0000
Epoch 41/100
3/3 [==============================] - 0s 125ms/step - loss: 0.0785 - accuracy: 1.0000 - val_loss: 0.0132 - val_accuracy: 1.0000
Epoch 42/100
3/3 [==============================] - 0s 122ms/step - loss: 0.0365 - accuracy: 1.0000 - val_loss: 0.0037 - val_accuracy: 1.0000
Epoch 43/100
3/3 [==============================] - 0s 132ms/step - loss: 7.2696e-04 - accuracy: 1.0000 - val_loss: 8.3408e-04 - val_accuracy: 1.0000
Epoch 44/100
3/3 [==============================] - 0s 132ms/step - loss: 2.0891e-04 - accuracy: 1.0000 - val_loss: 0.0047 - val_accuracy: 1.0000
Epoch 45/100
3/3 [==============================] - 0s 113ms/step - loss: 0.0067 - accuracy: 1.0000 - val_loss: 0.0139 - val_accuracy: 1.0000
Epoch 46/100
3/3 [==============================] - 0s 95ms/step - loss: 0.0029 - accuracy: 1.0000 - val_loss: 0.0545 - val_accuracy: 1.0000
Epoch 47/100
3/3 [==============================] - 0s 63ms/step - loss: 0.0829 - accuracy: 0.8889 - val_loss: 2.2798e-04 - val_accuracy: 1.0000
Epoch 48/100
3/3 [==============================] - 0s 64ms/step - loss: 0.0430 - accuracy: 1.0000 - val_loss: 0.1316 - val_accuracy: 1.0000
Epoch 49/100
3/3 [==============================] - 0s 73ms/step - loss: 0.0714 - accuracy: 1.0000 - val_loss: 0.0144 - val_accuracy: 1.0000
Epoch 50/100
3/3 [==============================] - 0s 65ms/step - loss: 0.0182 - accuracy: 1.0000 - val_loss: 0.0211 - val_accuracy: 1.0000
Epoch 51/100
3/3 [==============================] - 0s 105ms/step - loss: 0.0221 - accuracy: 1.0000 - val_loss: 0.0283 - val_accuracy: 1.0000
Epoch 52/100
3/3 [==============================] - 0s 60ms/step - loss: 0.0108 - accuracy: 1.0000 - val_loss: 0.0274 - val_accuracy: 1.0000
Epoch 53/100
3/3 [==============================] - 0s 68ms/step - loss: 0.0343 - accuracy: 1.0000 - val_loss: 0.0013 - val_accuracy: 1.0000
Epoch 54/100
3/3 [==============================] - 0s 72ms/step - loss: 1.3346e-04 - accuracy: 1.0000 - val_loss: 0.0839 - val_accuracy: 1.0000
Epoch 55/100
3/3 [==============================] - 0s 96ms/step - loss: 0.2268 - accuracy: 0.8571 - val_loss: 1.6563e-04 - val_accuracy: 1.0000
Epoch 56/100
3/3 [==============================] - 0s 69ms/step - loss: 0.0122 - accuracy: 1.0000 - val_loss: 0.6128 - val_accuracy: 0.8000
Epoch 57/100
3/3 [==============================] - 0s 71ms/step - loss: 0.0484 - accuracy: 1.0000 - val_loss: 0.0013 - val_accuracy: 1.0000
Epoch 58/100
3/3 [==============================] - 0s 64ms/step - loss: 2.9249e-04 - accuracy: 1.0000 - val_loss: 2.7412e-06 - val_accuracy: 1.0000
Epoch 59/100
3/3 [==============================] - 0s 73ms/step - loss: 0.0384 - accuracy: 1.0000 - val_loss: 2.9523e-05 - val_accuracy: 1.0000
Epoch 60/100
3/3 [==============================] - 0s 64ms/step - loss: 0.0434 - accuracy: 1.0000 - val_loss: 5.8240e-05 - val_accuracy: 1.0000
Epoch 61/100
3/3 [==============================] - 0s 96ms/step - loss: 5.5805e-05 - accuracy: 1.0000 - val_loss: 3.2121e-05 - val_accuracy: 1.0000
Epoch 62/100
3/3 [==============================] - 0s 69ms/step - loss: 6.3095e-05 - accuracy: 1.0000 - val_loss: 2.5540e-05 - val_accuracy: 1.0000
Epoch 63/100
3/3 [==============================] - 0s 73ms/step - loss: 7.3955e-05 - accuracy: 1.0000 - val_loss: 2.2462e-05 - val_accuracy: 1.0000
Epoch 64/100
3/3 [==============================] - 0s 68ms/step - loss: 0.0075 - accuracy: 1.0000 - val_loss: 2.3647e-05 - val_accuracy: 1.0000
Epoch 65/100
3/3 [==============================] - 0s 63ms/step - loss: 0.0014 - accuracy: 1.0000 - val_loss: 2.4683e-05 - val_accuracy: 1.0000
Epoch 66/100
3/3 [==============================] - 0s 65ms/step - loss: 1.5659e-04 - accuracy: 1.0000 - val_loss: 2.5196e-05 - val_accuracy: 1.0000
Epoch 67/100
3/3 [==============================] - 0s 73ms/step - loss: 6.2083e-06 - accuracy: 1.0000 - val_loss: 2.5498e-05 - val_accuracy: 1.0000
Epoch 68/100
3/3 [==============================] - 0s 64ms/step - loss: 1.2098e-05 - accuracy: 1.0000 - val_loss: 2.5702e-05 - val_accuracy: 1.0000
Epoch 69/100
3/3 [==============================] - 0s 66ms/step - loss: 2.6405e-05 - accuracy: 1.0000 - val_loss: 2.5765e-05 - val_accuracy: 1.0000
Epoch 70/100
3/3 [==============================] - 0s 102ms/step - loss: 1.3673e-05 - accuracy: 1.0000 - val_loss: 2.5645e-05 - val_accuracy: 1.0000
Epoch 71/100
3/3 [==============================] - 0s 64ms/step - loss: 1.1260e-04 - accuracy: 1.0000 - val_loss: 2.5483e-05 - val_accuracy: 1.0000
Epoch 72/100
3/3 [==============================] - 0s 75ms/step - loss: 7.4602e-06 - accuracy: 1.0000 - val_loss: 2.5221e-05 - val_accuracy: 1.0000
Epoch 73/100
3/3 [==============================] - 0s 68ms/step - loss: 4.0053e-05 - accuracy: 1.0000 - val_loss: 2.4882e-05 - val_accuracy: 1.0000
Epoch 74/100
3/3 [==============================] - 0s 111ms/step - loss: 1.7304e-05 - accuracy: 1.0000 - val_loss: 2.4589e-05 - val_accuracy: 1.0000
Epoch 75/100
3/3 [==============================] - 0s 59ms/step - loss: 1.8189e-05 - accuracy: 1.0000 - val_loss: 2.4353e-05 - val_accuracy: 1.0000
Epoch 76/100
3/3 [==============================] - 0s 70ms/step - loss: 2.5816e-04 - accuracy: 1.0000 - val_loss: 2.4040e-05 - val_accuracy: 1.0000
Epoch 77/100
3/3 [==============================] - 0s 70ms/step - loss: 3.6686e-06 - accuracy: 1.0000 - val_loss: 2.3747e-05 - val_accuracy: 1.0000
Epoch 78/100
3/3 [==============================] - 0s 102ms/step - loss: 5.8700e-05 - accuracy: 1.0000 - val_loss: 2.3453e-05 - val_accuracy: 1.0000
Epoch 79/100
3/3 [==============================] - 0s 67ms/step - loss: 1.9618e-04 - accuracy: 1.0000 - val_loss: 2.3165e-05 - val_accuracy: 1.0000
Epoch 80/100
3/3 [==============================] - 0s 69ms/step - loss: 4.9517e-06 - accuracy: 1.0000 - val_loss: 2.2918e-05 - val_accuracy: 1.0000
Epoch 81/100
3/3 [==============================] - 0s 98ms/step - loss: 4.6140e-05 - accuracy: 1.0000 - val_loss: 2.2658e-05 - val_accuracy: 1.0000
Epoch 82/100
3/3 [==============================] - 0s 108ms/step - loss: 3.1362e-05 - accuracy: 1.0000 - val_loss: 2.2340e-05 - val_accuracy: 1.0000
Epoch 83/100
3/3 [==============================] - 0s 65ms/step - loss: 7.9430e-06 - accuracy: 1.0000 - val_loss: 2.2109e-05 - val_accuracy: 1.0000
Epoch 84/100
3/3 [==============================] - 0s 73ms/step - loss: 3.1944e-06 - accuracy: 1.0000 - val_loss: 2.1930e-05 - val_accuracy: 1.0000
Epoch 85/100
3/3 [==============================] - 0s 66ms/step - loss: 2.8734e-05 - accuracy: 1.0000 - val_loss: 2.1752e-05 - val_accuracy: 1.0000
Epoch 86/100
3/3 [==============================] - 0s 67ms/step - loss: 1.3287e-04 - accuracy: 1.0000 - val_loss: 2.1501e-05 - val_accuracy: 1.0000
Epoch 87/100
3/3 [==============================] - 0s 112ms/step - loss: 1.0672e-04 - accuracy: 1.0000 - val_loss: 2.1197e-05 - val_accuracy: 1.0000
Epoch 88/100
3/3 [==============================] - 0s 120ms/step - loss: 8.2121e-05 - accuracy: 1.0000 - val_loss: 2.0926e-05 - val_accuracy: 1.0000
Epoch 89/100
3/3 [==============================] - 0s 102ms/step - loss: 1.0130e-05 - accuracy: 1.0000 - val_loss: 2.0701e-05 - val_accuracy: 1.0000
Epoch 90/100
3/3 [==============================] - 0s 101ms/step - loss: 3.7200e-06 - accuracy: 1.0000 - val_loss: 2.0527e-05 - val_accuracy: 1.0000
Epoch 91/100
3/3 [==============================] - 0s 124ms/step - loss: 1.9436e-05 - accuracy: 1.0000 - val_loss: 2.0297e-05 - val_accuracy: 1.0000
Epoch 92/100
3/3 [==============================] - 0s 110ms/step - loss: 8.6931e-05 - accuracy: 1.0000 - val_loss: 2.0080e-05 - val_accuracy: 1.0000
Epoch 93/100
3/3 [==============================] - 0s 113ms/step - loss: 1.9012e-05 - accuracy: 1.0000 - val_loss: 1.9757e-05 - val_accuracy: 1.0000
Epoch 94/100
3/3 [==============================] - 0s 132ms/step - loss: 3.7882e-05 - accuracy: 1.0000 - val_loss: 1.9481e-05 - val_accuracy: 1.0000
Epoch 95/100
3/3 [==============================] - 0s 71ms/step - loss: 5.5576e-06 - accuracy: 1.0000 - val_loss: 1.9264e-05 - val_accuracy: 1.0000
Epoch 96/100
3/3 [==============================] - 0s 63ms/step - loss: 3.0488e-05 - accuracy: 1.0000 - val_loss: 1.9071e-05 - val_accuracy: 1.0000
Epoch 97/100
3/3 [==============================] - 0s 73ms/step - loss: 1.9546e-05 - accuracy: 1.0000 - val_loss: 1.8843e-05 - val_accuracy: 1.0000
Epoch 98/100
3/3 [==============================] - 0s 98ms/step - loss: 1.8926e-05 - accuracy: 1.0000 - val_loss: 1.8620e-05 - val_accuracy: 1.0000
Epoch 99/100
3/3 [==============================] - 0s 63ms/step - loss: 1.7655e-05 - accuracy: 1.0000 - val_loss: 1.8338e-05 - val_accuracy: 1.0000
Epoch 100/100
3/3 [==============================] - 0s 64ms/step - loss: 6.2897e-06 - accuracy: 1.0000 - val_loss: 1.8099e-05 - val_accuracy: 1.0000
<keras.callbacks.History at 0x7bc4905ae680>
```

## Evaluating The Model
Once our model is fully trained, we can put it up to the test! It's essential to evaluate the model's performance on unseen data to understand how well the model trained.
The following code demonstrates how to evaluate our trained model using a set of test images:
``` python
dir_path = '/content/happy vs not happy/testing'

for i in os.listdir(dir_path):
    img = image.load_img(os.path.join(dir_path, i), target_size=(250, 250))
    plt.imshow(img)
    plt.show()
    
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)  # Expand dimensions ONCE
    
    val = model.predict(X)
    
    if val == 0:
        print('Not Happy!')
    else:
        print('Happy!')
```
We takes the images from the given directory and plot it along with converting the image to a numpy array + expanding its dimensions to match the expected shape for model prediction

![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/d7284397-0d1f-442e-a2f8-5df6076d6e26)

![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/7438c420-5c39-42bc-be2a-c5d3f1ddcfb4)

![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/f1d6ab23-72dd-486d-bef9-eae79eb08326)
