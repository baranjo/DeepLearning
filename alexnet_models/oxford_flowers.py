from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import time
import os

t0 = time.time()
data_dir = "../datasets/oxford_flowers"
batch_size = 64
img_height = 224
img_width = 224
seed = 123
validation_data = 0.1

train_ds = keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  label_mode='int',
  validation_split=validation_data,
  subset="training",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

test_ds = keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  label_mode='int',
  validation_split=validation_data,
  subset="validation",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

num_classes = len(os.listdir(data_dir))

opt = keras.optimizers.Adam(learning_rate=0.001)

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
  optimizer=opt,
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'])

# Calculate the number of batches in the train_ds dataset
num_batches = tf.data.experimental.cardinality(train_ds).numpy()

# Train the model using a single model.fit() call
model.fit(
    train_ds,
#    steps_per_epoch=num_batches,
    epochs=50,
    validation_data=test_ds,
#    validation_steps=test_ds.cardinality().numpy()
)

# Save the model
# doesn't work for tensorflow v2.6.0
#model.save_weights('saved_models/oxford_flowers/weights.h5')
#model.save('saved_models/oxford_flowers/model')

# Evaluate the model
model.evaluate(test_ds)

t1 = time.time() - t0
secs = int(t1 % 60)
mins = int(t1 // 60)
print(f"Runtime: {mins} minutes {secs} second(s)")