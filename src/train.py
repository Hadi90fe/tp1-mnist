import tensorflow as tf
import os
from model import create_model
from utils import save_training_curves, log_training

# Charger MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalisation et ajout du canal (28x28x1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

if __name__ == "__main__":
    # Créer le modèle
    model = create_model(learning_rate=0.01)
    print(model.summary())

    # Entraînement
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=25,
        validation_split=0.1,
        shuffle=True
    )

    # Sauvegarde du modèle
    os.makedirs("data", exist_ok=True)
    model.save("data/tp1.keras")
    print("✅ Modèle sauvegardé sous data/tp1.keras")

    # Sauvegarder graphiques et logs
    save_training_curves(history, output_dir="data/output/graphs")
    log_training(history, output_path="data/output/logs/training_log.txt")

    # Évaluation finale
    print("\n📊 Évaluation sur le jeu d'entraînement :")
    model.evaluate(x_train, y_train, batch_size=128, verbose=2)
    print("\n📊 Évaluation sur le jeu de test :")
    model.evaluate(x_test, y_test, batch_size=128, verbose=2)
