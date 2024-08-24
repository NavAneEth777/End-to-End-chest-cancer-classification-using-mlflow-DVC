import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path



class Training:
    def __init__(self, config):
        self.config = config
        self.model = None  # Initialize the model attribute
        self.optimizer = None

    def get_base_model(self):
        # Example model creation
        base_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False)
        self.model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')  # Adjust based on number of classes
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def compile_model(self):
        if self.model is None:
            raise ValueError("Model is not initialized. Call get_base_model() first.")
        
        # Compile the model with a fresh optimizer
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_valid_generator(self):
        # Code to initialize train and valid generators
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode='categorical',  # Ensures targets are categorical
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
        

    def train(self):
        if self.model is None:
            raise ValueError("Model is not initialized. Call get_base_model() first.")
        
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized. Call compile_model() first.")
        
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        # Save the model after training
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

    def save_model(self, path, model):
        """Save the trained model to the specified path."""
        model.save(path)
        print(f"Model saved at: {path}")

    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None

    def get_base_model(self):
        # Example model creation
        base_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False)
        self.model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')  # Adjust based on number of classes
        ])

    def compile_model(self):
        # Recreate the optimizer each time the model is compiled
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Compile the model with the new optimizer
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        if self.model is None:
            raise ValueError("Model is not initialized. Call get_base_model() first.")
        
        # Recompile model to ensure compatibility
        self.compile_model()

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        # Save the model after training
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

    def save_model(self, path, model):
        """Save the trained model to the specified path."""
        model.save(path)
        print(f"Model saved at: {path}")
