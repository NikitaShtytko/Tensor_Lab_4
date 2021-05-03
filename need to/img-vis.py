import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import glob
import pathlib
from datetime import datetime
import random

data_dir = pathlib.Path('../food-101/images/')

logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")

file_writer = tf.summary.create_file_writer(logdir)
    
all_images = list(data_dir.glob('*/*'))
images = []
for i in range(0, 6):
	a = PIL.Image.open(str(random.choice(all_images)))

	a = a.resize((224, 224), PIL.Image.ANTIALIAS)

	pix = np.array(a)

	img = np.reshape(pix, (224, 224, 3))
	images.append(img)

ar = np.array(images)

ar = tf.keras.layers.experimental.preprocessing.RandomRotation(0.5, fill_mode='wrap', interpolation='nearest')(ar)

with file_writer.as_default():
  	tf.summary.image("Training data", ar, step=1, max_outputs=15)
