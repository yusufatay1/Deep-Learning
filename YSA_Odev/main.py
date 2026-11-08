
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# =====================================================
# 1. GENEL AYARLAR
# =====================================================

SEED = 23
EPOCHS = 300
BATCH_SIZE = 200

OUTPUT_DIR = "sonuclar"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# =====================================================
# 2. VERİ SETİ
# =====================================================

p = np.linspace(-2, 2, 200).reshape(-1, 1).astype(np.float32)

# Hedef denklem
y = (1 + np.sin(np.pi * p / 4)).astype(np.float32)


# =====================================================
# 3. R2 CALLBACK
# =====================================================

class R2Callback(tf.keras.callbacks.Callback):

    def __init__(self, x, y_true):
        super().__init__()
        self.x = x
        self.y_true = y_true
        self.r2_values = []

    def on_epoch_end(self, epoch, logs=None):

        predictions = self.model(
            self.x, training=False
        ).numpy()

        ss_res = np.sum((self.y_true - predictions) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)

        r2 = 1 - ss_res / ss_tot

        self.r2_values.append(float(r2))


# =====================================================
# 4. MODEL OLUŞTURMA
# =====================================================

def build_model(
    activation="relu",
    learning_rate=0.1,
    optimizer_name="SGD",
    initializer="random_normal",
    hidden_layers=1,
    neurons=2
):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(1,)))

    # Gizli katmanlar
    for _ in range(hidden_layers):

        model.add(
            tf.keras.layers.Dense(
                neurons,
                kernel_initializer=initializer
            )
        )

        if activation == "LeakyReLU":

            model.add(
                tf.keras.layers.LeakyReLU(
                    negative_slope=0.01
                )
            )

        else:

            model.add(
                tf.keras.layers.Activation(activation)
            )

    # Çıkış katmanı
    model.add(
        tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer=initializer
        )
    )

    # Optimizasyon algoritması
    if optimizer_name == "SGD":

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate
        )

    elif optimizer_name == "RMSProp":

        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate
        )

    elif optimizer_name == "Adam":

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

    model.compile(
        optimizer=optimizer,
        loss="mse"
    )

    return model


# =====================================================
# 5. ÜÇ GRAFİĞİ TEK GÖRSELDE OLUŞTURMA
# =====================================================

def save_graphs(experiment_name, best_result):

    safe_name = experiment_name.lower().replace(" ", "_")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Grafik 1: Epoch / MSE
    axes[0].plot(best_result["loss"], color="red")

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (Logaritmik)")
    axes[0].set_title("Epoch / MSE Kaybı")
    axes[0].grid(True)

    # Grafik 2: Hedef / Ağ Çıkışı
    axes[1].plot(
        p,
        y,
        label="Hedef",
        color="blue"
    )

    axes[1].plot(
        p,
        best_result["predictions"],
        label="Ağ Çıkışı",
        color="red",
        linestyle="--"
    )

    axes[1].set_xlabel("p (Girdi)")
    axes[1].set_ylabel("y (Çıktı)")

    axes[1].set_title(
        f"Hedef ve Ağ Çıkışı (R2 = {best_result['metrics']['R2']:.4f})"
    )

    axes[1].legend()
    axes[1].grid(True)

    # Grafik 3: Epoch / R2
    axes[2].plot(
        best_result["r2_history"],
        color="green"
    )

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R2 Skoru")
    axes[2].set_title("Epoch / R2 Skoru Değişimi")
    axes[2].grid(True)

    fig.suptitle(experiment_name, fontsize=14)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, safe_name + ".png"),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# 6. DENEY ÇALIŞTIRMA FONKSİYONU
# =====================================================

def run_experiment(experiment_name, parameter_name, values, base_settings):

    results = []

    best_result = None
    best_r2 = -float("inf")

    print("\n" + "=" * 65)
    print(experiment_name)
    print("=" * 65)

    for value in values:

        # Her model için rastgelelik başlangıcını sabitle
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        settings = base_settings.copy()
        settings[parameter_name] = value

        model = build_model(**settings)

        r2_callback = R2Callback(p, y)

        start_time = time.perf_counter()

        history = model.fit(
            p,
            y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[r2_callback]
        )

        training_time = time.perf_counter() - start_time

        predictions = model(
            p,
            training=False
        ).numpy()

        # Performans hesaplamaları
        mse = float(np.mean((y - predictions) ** 2))
        rmse = float(np.sqrt(mse))

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = float(1 - ss_res / ss_tot)

        result = {
            "Değer": str(value),
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Süre": training_time
        }

        results.append(result)

        print(
            f"{parameter_name}: {value} | "
            f"MSE: {mse:.6f} | "
            f"RMSE: {rmse:.6f} | "
            f"R2: {r2:.6f} | "
            f"Süre: {training_time:.2f} saniye"
        )

        # En yüksek R2 değerine sahip modeli sakla
        if np.isfinite(r2) and r2 > best_r2:

            best_r2 = r2

            best_result = {
                "value": value,
                "predictions": predictions.copy(),
                "loss": history.history["loss"].copy(),
                "r2_history": r2_callback.r2_values.copy(),
                "metrics": result.copy()
            }

    # Deneyin en iyi modeline ait grafikleri oluştur
    if best_result is not None:

        save_graphs(experiment_name, best_result)

        print("\nEn başarılı değer:", best_result["value"])
        print("En yüksek R2:", round(best_r2, 6))

    else:

        print("\nBu deneyde geçerli sonuç elde edilemedi.")

    return results


# =====================================================
# 7. TEMEL MODEL AYARLARI
# =====================================================

base_settings = {

    "activation": "relu",
    "learning_rate": 0.1,
    "optimizer_name": "SGD",
    "initializer": "random_normal",
    "hidden_layers": 1,
    "neurons": 2

}


# =====================================================
# 8. DENEY 1 - AKTİVASYON FONKSİYONLARI
# =====================================================

activation_results = run_experiment(
    "Deney 1 Aktivasyon",
    "activation",
    [
        "linear",
        "sigmoid",
        "tanh",
        "relu",
        "LeakyReLU",
        "elu",
        "silu"
    ],
    base_settings
)


# =====================================================
# 9. DENEY 2 - ÖĞRENME ORANI
# =====================================================

learning_rate_results = run_experiment(
    "Deney 2 Ogrenme Orani",
    "learning_rate",
    [0.1, 0.01, 0.001],
    base_settings
)


# =====================================================
# 10. DENEY 3 - OPTİMİZASYON ALGORİTMASI
# =====================================================

optimizer_results = run_experiment(
    "Deney 3 Optimizasyon",
    "optimizer_name",
    ["SGD", "RMSProp", "Adam"],
    base_settings
)


# =====================================================
# 11. DENEY 4 - BAŞLANGIÇ DAĞILIMI
# =====================================================

initializer_results = run_experiment(
    "Deney 4 Baslangic Dagilimi",
    "initializer",
    ["random_normal", "random_uniform"],
    base_settings
)


# =====================================================
# 12. DENEY 5 - GİZLİ KATMAN SAYISI
# =====================================================

layer_results = run_experiment(
    "Deney 5 Gizli Katman",
    "hidden_layers",
    [1, 2, 5, 10, 20],
    base_settings
)


# =====================================================
# 13. DENEY 6 - NÖRON SAYISI
# =====================================================

neuron_results = run_experiment(
    "Deney 6 Noron Sayisi",
    "neurons",
    [2, 5, 10, 20, 50],
    base_settings
)


# =====================================================
# 14. SONUÇ TABLOLARINI TERMİNALE YAZDIRMA
# =====================================================

all_results = [
    ("Deney 1 - Aktivasyon", activation_results),
    ("Deney 2 - Öğrenme Oranı", learning_rate_results),
    ("Deney 3 - Optimizasyon", optimizer_results),
    ("Deney 4 - Başlangıç Dağılımı", initializer_results),
    ("Deney 5 - Gizli Katman", layer_results),
    ("Deney 6 - Nöron Sayısı", neuron_results)
]

for experiment_name, results in all_results:

    print("\n" + "=" * 85)
    print(experiment_name)
    print("=" * 85)

    print(
        f"{'Değer':<20} {'MSE':<15} {'RMSE':<15} "
        f"{'R2':<15} {'Süre (sn)':<15}"
    )

    for result in results:

        print(
            f"{result['Değer']:<20} "
            f"{result['MSE']:<15.6f} "
            f"{result['RMSE']:<15.6f} "
            f"{result['R2']:<15.6f} "
            f"{result['Süre']:<15.2f}"
        )


# =====================================================
# 15. TAMAMLANDI
# =====================================================

print("\n" + "=" * 65)
print("BÜTÜN DENEYLER TAMAMLANDI!")
print("=" * 65)

print("Toplam deney sayısı: 6")
print("Toplam eğitilen model sayısı: 25")
print("Toplam grafik görseli: 6")
print("Grafikler şu klasöre kaydedildi:", OUTPUT_DIR)
