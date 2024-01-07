import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

# Define the base directory for the dataset
base_dir = r'C:\Users\25470\Desktop\Dataset'

# Define subdirectories for train, validation, and test datasets
train_dir = os.path.join(base_dir, r'C:\Users\25470\Desktop\Dataset\Train')
val_dir = os.path.join(base_dir, r'C:\Users\25470\Desktop\Dataset\val')
test_dir = os.path.join(base_dir, r'C:\Users\25470\Desktop\Dataset\Test')

# Set up data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',  # Use 'sparse' for class indices
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',  # Use 'sparse' for class indices
    shuffle=False
)

# Build the model
base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(3, activation='softmax')(x)

model = models.Model(base_model.input, x)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=2,
    validation_data=validation_generator
)

# Save the trained model
model.save('C:\\Users\\25470\\PycharmProjects\\myproject\\model\\trainedmodel.keras')
