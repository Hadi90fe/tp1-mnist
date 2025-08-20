import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Charger modèle
model = tf.keras.models.load_model("data/tp1.keras")

# Charger MNIST
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]

# Indices demandés
indices = [1, 6, 3513, 10123, 43213]

output_dir = "data/output/predictions"
os.makedirs(output_dir, exist_ok=True)

for idx in indices:
    img = x_train[idx]
    true = y_train[idx]

    # Prédiction
    pred = model.predict(img[tf.newaxis, ...], verbose=0)
    pred_label = pred.argmax()

    # Sauvegarder image prédite
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Vrai: {true} — Prédit: {pred_label}")
    plt.savefig(f"{output_dir}/prediction_{idx}.png")
    plt.close()

    print(f"Image {idx} → Vrai: {true} | Prédit: {pred_label}")
