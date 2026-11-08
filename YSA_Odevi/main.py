# Bir Regresyon Probleminde Ağ Meta Parametrelerinin Ağ Performansına Etkisi
# Hedef: y = 1 + sin(pi * p / 4),  -2 <= p <= 2
#
# Başlangıç ağı : 1 gizli katman, 2 nöron (ReLU), 1 lineer çıkış
# Her model 300 epoch çalıştırılır. Her tabloda sadece ilgili meta parametre
# değiştirilir, diğerleri başlangıç (taban) ayarında sabit tutulur.

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from keras import layers

# ---------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------
SEED = 23         # okul numarasının son 2 hanesi
EPOCH = 300
NOKTA_SAYISI = 21

keras.utils.set_random_seed(SEED)

# ---------------------------------------------------------------
# VERİ
# ---------------------------------------------------------------
p = np.linspace(-2, 2, NOKTA_SAYISI).reshape(-1, 1)
y = 1 + np.sin(np.pi * p / 4)
y_varyans = np.var(y)


# ---------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------------
def aktivasyon_ekle(model, ad):
    if ad == "Lineer":
        model.add(layers.Activation("linear"))
    elif ad == "Sigmoid":
        model.add(layers.Activation("sigmoid"))
    elif ad == "Hiperbolik Tanjant":
        model.add(layers.Activation("tanh"))
    elif ad == "ReLU":
        model.add(layers.Activation("relu"))
    elif ad == "LeakyReLU":
        model.add(layers.LeakyReLU())
    elif ad == "ELU":
        model.add(layers.Activation("elu"))
    elif ad == "SiLU":
        model.add(layers.Activation("silu"))


def baslatici_olustur(dagilim, seed):
    if dagilim == "normal":
        return keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=seed)
    else:
        return keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=seed)


def optimizer_olustur(ad, ogrenme_orani):
    if ad == "SGD":
        return keras.optimizers.SGD(learning_rate=ogrenme_orani)
    elif ad == "RMSProp":
        return keras.optimizers.RMSprop(learning_rate=ogrenme_orani)
    else:
        return keras.optimizers.Adam(learning_rate=ogrenme_orani)


def model_olustur(aktivasyon, ogrenme_orani, optimizer_ad, dagilim,
                  gizli_katman, noron):
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential()
    model.add(keras.Input(shape=(1,)))
    for k in range(gizli_katman):
        model.add(layers.Dense(noron,
                               kernel_initializer=baslatici_olustur(dagilim, SEED + k)))
        aktivasyon_ekle(model, aktivasyon)
    model.add(layers.Dense(1, activation="linear",
                           kernel_initializer=baslatici_olustur(dagilim, SEED + 100)))
    model.compile(loss="mse", optimizer=optimizer_olustur(optimizer_ad, ogrenme_orani))
    return model


def calistir(aktivasyon, ogrenme_orani, optimizer_ad, dagilim, gizli_katman, noron):
    """Modeli eğitir; mse, rmse, r2, sure, mse_gecmisi, cikis döndürür."""
    model = model_olustur(aktivasyon, ogrenme_orani, optimizer_ad, dagilim,
                          gizli_katman, noron)
    baslangic = time.time()
    gecmis = model.fit(p, y, epochs=EPOCH, batch_size=NOKTA_SAYISI,
                       shuffle=False, verbose=0)
    sure = time.time() - baslangic

    cikis = model.predict(p, verbose=0)
    mse = np.mean((y - cikis) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - mse / y_varyans
    mse_gecmisi = np.array(gecmis.history["loss"])
    return mse, rmse, r2, sure, mse_gecmisi, cikis


def grafik_ciz(deney_no, baslik, mse_gecmisi, cikis):
    epoch_ekseni = np.arange(1, EPOCH + 1)
    r2_gecmisi = 1 - mse_gecmisi / y_varyans

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.semilogy(epoch_ekseni, mse_gecmisi)
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log)")
    plt.title("Epoch / MSE Hatası")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(p, y, label="Hedef")
    plt.plot(p, cikis, "--", label="Ağ çıkışı")
    plt.xlabel("p")
    plt.ylabel("y")
    plt.title("Hedef - Çıkış")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epoch_ekseni, r2_gecmisi)
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.title("Epoch / R2")
    plt.grid(True)

    plt.suptitle("Deney " + str(deney_no) + " - en iyi sonuç: " + baslik)
    plt.tight_layout()
    plt.savefig("grafik_deney" + str(deney_no) + ".png", dpi=150)
    plt.close()


def deney_yap(deney_no, deney_adi, degerler, aktivasyon, ogrenme_orani,
              optimizer_ad, dagilim, gizli_katman, noron):
    """
    'degerler' dizisindeki her değeri, deney_adi'na göre ilgili parametreye
    atayıp modeli çalıştırır. En iyi (en yüksek R2) değerin indeksini döndürür.
    """
    n = len(degerler)
    mse_dizi = np.zeros(n)
    rmse_dizi = np.zeros(n)
    r2_dizi = np.zeros(n)
    sure_dizi = np.zeros(n)
    gecmis_dizi = np.zeros((n, EPOCH))
    cikis_dizi = np.zeros((n, NOKTA_SAYISI))

    for i in range(n):
        a = aktivasyon
        lr = ogrenme_orani
        opt = optimizer_ad
        d = dagilim
        gk = gizli_katman
        nr = noron

        if deney_adi == "aktivasyon":
            a = degerler[i]
        elif deney_adi == "ogrenme_orani":
            lr = float(degerler[i])
        elif deney_adi == "optimizer":
            opt = degerler[i]
        elif deney_adi == "dagilim":
            d = degerler[i]
        elif deney_adi == "gizli_katman":
            gk = int(degerler[i])
        elif deney_adi == "noron":
            nr = int(degerler[i])

        mse, rmse, r2, sure, gecmis, cikis = calistir(a, lr, opt, d, gk, nr)
        mse_dizi[i] = mse
        rmse_dizi[i] = rmse
        r2_dizi[i] = r2
        sure_dizi[i] = sure
        gecmis_dizi[i, :] = gecmis
        cikis_dizi[i, :] = cikis[:, 0]

    # En iyi sonucu bul (NaN / sonsuz sonuçlar dikkate alınmaz)
    en_iyi = 0
    en_iyi_r2 = -np.inf
    for i in range(n):
        if np.isfinite(r2_dizi[i]) and r2_dizi[i] > en_iyi_r2:
            en_iyi_r2 = r2_dizi[i]
            en_iyi = i

    # Tabloyu yazdır (en iyi satır ** ile işaretli)
    print()
    print("Tablo " + str(deney_no) + ". " + deney_adi + " değişimi")
    print("(Öğrenme oranı: " + str(ogrenme_orani) + ", Yöntem: " + optimizer_ad +
          ", Dağılım: " + dagilim + ", Gizli katman: " + str(gizli_katman) +
          ", Nöron: " + str(noron) + ", Aktivasyon: " + aktivasyon + ")")
    print("%-22s %12s %12s %12s %10s" % ("", "MSE", "RMSE", "R2", "Süre (sn)"))
    for i in range(n):
        satir = "%-22s %12.6f %12.6f %12.6f %10.2f" % (
            str(degerler[i]), mse_dizi[i], rmse_dizi[i], r2_dizi[i], sure_dizi[i])
        if i == en_iyi:
            satir = satir + "   <-- **EN İYİ**"
        print(satir)

    # En iyi sonuç için grafikler
    grafik_ciz(deney_no, str(degerler[en_iyi]), gecmis_dizi[en_iyi, :],
               cikis_dizi[en_iyi, :].reshape(-1, 1))

    return en_iyi


# ---------------------------------------------------------------
# DENEYLER
# ---------------------------------------------------------------
aktivasyonlar = np.array(["Lineer", "Sigmoid", "Hiperbolik Tanjant", "ReLU",
                          "LeakyReLU", "ELU", "SiLU"])
ogrenme_oranlari = np.array([0.1, 0.01, 0.001])
optimizerler = np.array(["SGD", "RMSProp", "Adam"])
dagilimlar = np.array(["normal", "uniform"])
gizli_katmanlar = np.array([1, 2, 5, 10, 20])
noronlar = np.array([2, 5, 10, 20, 50])

# Taban ayarlar (tüm tablolarda sabit kalır, sadece incelenen parametre değişir)
aktivasyon = "ReLU"
ogrenme_orani = 0.1
optimizer_ad = "SGD"
dagilim = "normal"
gizli_katman = 1
noron = 2

# Tablo 1: Aktivasyon fonksiyonları
deney_yap(1, "aktivasyon", aktivasyonlar, aktivasyon, ogrenme_orani,
          optimizer_ad, dagilim, gizli_katman, noron)

# Tablo 2: Öğrenme oranı
deney_yap(2, "ogrenme_orani", ogrenme_oranlari, aktivasyon, ogrenme_orani,
          optimizer_ad, dagilim, gizli_katman, noron)

# Tablo 3: Öğrenme oranı ayar yöntemi
deney_yap(3, "optimizer", optimizerler, aktivasyon, ogrenme_orani,
          optimizer_ad, dagilim, gizli_katman, noron)

# Tablo 4: Başlangıç değerlerinin olasılık dağılımı
deney_yap(4, "dagilim", dagilimlar, aktivasyon, ogrenme_orani,
          optimizer_ad, dagilim, gizli_katman, noron)

# Tablo 5: Gizli katman sayısı
deney_yap(5, "gizli_katman", gizli_katmanlar, aktivasyon, ogrenme_orani,
          optimizer_ad, dagilim, gizli_katman, noron)

# Tablo 6: Gizli katmandaki nöron sayısı
deney_yap(6, "noron", noronlar, aktivasyon, ogrenme_orani,
          optimizer_ad, dagilim, gizli_katman, noron)
