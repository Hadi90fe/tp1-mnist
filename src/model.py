import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(learning_rate=0.01):
    """
    Crée et compile un modèle CNN adapté au dataset MNIST.
    Inspiré de l'exemple officiel Keras.
    source: https://keras.io/examples/vision/mnist_convnet/
    """
    model = models.Sequential([
        # Première couche conv → extraction de motifs basiques
        layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D(pool_size=(2,2)),

        # Deuxième couche conv → motifs plus complexes
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),

        # Régularisation
        layers.Dropout(0.25),
        layers.Flatten(),

        # Couche dense intermédiaire
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        # Couche de sortie softmax (10 classes)
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
