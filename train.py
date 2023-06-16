import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from model import SR_CNN


#Download and Read
ds = tfds.load('div2k', split='train', shuffle_files=True, download=False) #download=True to download div2k dataset
ds = ds.map(lambda x: (tf.cast(x['lr'], tf.float32), tf.cast(x['hr'], tf.float32)))


sr_cnn = SR_CNN()

sr_cnn.compile(optimizer='adam',
              loss='MSE',
              metrics=tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"))

loss_his = []
sr_cnn.fit(ds, batch_size=1, epochs=1)