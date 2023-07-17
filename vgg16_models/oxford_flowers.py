import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.regularizers import l1, l2
import time
import os

t0 = time.time()

def count_images(directory):
  image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more if needed
  count = 0

  for root, _, files in os.walk(directory):
    for filename in files:
      if any(filename.lower().endswith(ext) for ext in image_extensions):
        count += 1

  return count


# Example usage
data_dir = "../datasets/oxford_flowers"
image_count = count_images(data_dir)
print(f"Number of images in the folder: {image_count}")

batch_size = 64
img_height = 224
img_width = 224

seed = 123
validation_data = 0.1

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  label_mode='int',
  validation_split=validation_data,
  subset="training",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  label_mode='int',
  validation_split=validation_data,
  subset="validation",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(os.listdir(data_dir))

# Load the pre-trained VGG-16 model without the top classification layer
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the weights of the pre-trained layers
for layer in vgg16.layers[:14]:
    layer.trainable = False

# Create a new model and add the pre-trained VGG-16 as a base
model = Sequential()
model.add(vgg16)

# Add custom fully connected layers on top of the VGG-16 base
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_regularizer=l1(0.01)))
model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(num_classes, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(
  optimizer=opt,
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'])

# Train the model
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=25,
  batch_size=batch_size
)

# Save the model
# doesn't work for tensorflow v2.6.0
#model.save_weights('saved_models/oxford_flowers/weights.h5')
#model.save('saved_models/oxford_flowers/model')

# Evaluate the model
model.evaluate(val_ds)

t1 = time.time() - t0
secs = int(t1 % 60)
mins = int(t1 // 60)
print(f"Runtime: {mins} minutes {secs} second(s)")
