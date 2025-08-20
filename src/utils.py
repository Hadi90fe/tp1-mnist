import matplotlib.pyplot as plt
import pandas as pd
import os

def save_training_curves(history, output_dir="data/output/graphs"):
    """
    Sauvegarde les courbes accuracy et loss en PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    hist_df = pd.DataFrame(history.history)

    # Accuracy
    plt.figure()
    plt.plot(hist_df['accuracy'], label='train acc')
    plt.plot(hist_df['val_accuracy'], label='val acc')
    plt.title("Accuracy au cours des epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()

    # Loss
    plt.figure()
    plt.plot(hist_df['loss'], label='train loss')
    plt.plot(hist_df['val_loss'], label='val loss')
    plt.title("Loss au cours des epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

def log_training(history, output_path="data/output/logs/training_log.txt"):
    """
    Sauvegarde les logs d'entraînement (accuracy & loss par epoch).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(output_path, index=False)
    print(f"✅ Logs sauvegardés dans {output_path}")
