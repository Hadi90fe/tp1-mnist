import unittest
import tensorflow as tf
import numpy as np

class TestMNISTModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Charger une seule fois le modèle pour tous les tests."""
        cls.model = tf.keras.models.load_model("data/tp1.keras")

    def test_output_shape(self):
        """
        Vérifie que la sortie du modèle est bien un vecteur de 10 classes (softmax).
        """
        x = np.random.rand(1, 28, 28, 1).astype("float32")
        y = self.model.predict(x, verbose=0)
        self.assertEqual(y.shape, (1, 10), "La sortie doit être de dimension (1,10)")

    def test_prediction_range(self):
        """
        Vérifie que les probabilités de sortie sont comprises entre 0 et 1.
        """
        x = np.random.rand(1, 28, 28, 1).astype("float32")
        y = self.model.predict(x, verbose=0)
        self.assertTrue((y >= 0).all() and (y <= 1).all(),
                        "Toutes les probabilités doivent être entre 0 et 1")

    def test_minimum_accuracy(self):
        """
        Vérifie que l'accuracy sur le jeu de test atteint un seuil minimum (ex: 90%).
        """
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.astype("float32") / 255.0
        x_test = x_test[..., tf.newaxis]

        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        self.assertGreater(acc, 0.90,
                           f"L'accuracy du modèle est trop faible ({acc:.2f}), minimum attendu: 0.90")

if __name__ == "__main__":
    unittest.main()
