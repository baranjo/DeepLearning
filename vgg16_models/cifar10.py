
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow import keras
import os
import tensorflow as tf
import time

t0 = time.time()

# Resize the images
new_size = (224, 224)  # Specify the new size you want for the images

def data_generator(images, labels, batch_size):
    num_samples = images.shape[0]
    num_batches = num_samples // batch_size

    while True:
        for i in range(num_batches):
            batch_images = images[i * batch_size : (i + 1) * batch_size]
            batch_labels = labels[i * batch_size : (i + 1) * batch_size]

            # Perform any preprocessing on the batch here
            batch_images = preprocess_input(batch_images)
            batch_images = tf.image.resize(batch_images, new_size)

            yield batch_images, batch_labels


cifar10 = keras.datasets.cifar10
(train_ds, train_labels), (test_ds, test_labels) = cifar10.load_data()

num_classes = 10
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

batch_size = 32
train_generator = data_generator(train_ds, train_labels, batch_size)
test_generator = data_generator(test_ds, test_labels, batch_size)

# Load the pre-trained VGG-16 model without the top classification layer
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the weights of the pre-trained layers
for layer in vgg16.layers:
    layer.trainable = False

# Create a new model and add the pre-trained VGG-16 as a base
model = Sequential()
model.add(vgg16)

# Add custom fully connected layers on top of the VGG-16 base
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator,
          steps_per_epoch=train_ds.shape[0] // batch_size,
          epochs=10,
          validation_data=test_generator,
          validation_steps=test_ds.shape[0] // batch_size)

model.evaluate(test_generator,
               steps=test_ds.shape[0] // batch_size)

# Save the model
# doesn't work for tensorflow v2.6.0
#model.save_weights('saved_models/cifar10_weights.h5')
#model.save('saved_models/cifar10')

t1 = time.time() - t0
secs = int(t1 % 60)
mins = int(t1 // 60)
print(f"Runtime: {mins} minutes {secs} second(s)")
