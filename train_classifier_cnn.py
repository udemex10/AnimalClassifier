import tensorflow as tf
from build_classifier_cnn import build_model

# Define constants
DATA_DIR = 'Dataset'  # replace with your path
BATCH_SIZE = 32
IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASS_NAMES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# Create ImageDataGenerator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Prepare training and validation datasets
train_ds = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode='categorical',
    subset='training')

val_ds = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode='categorical',
    subset='validation')

# Load model
model = tf.keras.models.load_model('model.h5')

# Train model
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save trained model
model.save('trained_model.h5')
