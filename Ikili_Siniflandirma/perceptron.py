
# Basit Ikili Siniflandirici (Perceptron)

# AND veri seti: [girdi1, girdi2, beklenen sonuc]
data = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]

# Elle belirlenen baslangic agirliklari ve bias
w1 = 0.2
w2 = -0.1
bias = -0.2

learning_rate = 0.1

batch_size = len(data)
epoch_size = batch_size

epoch = 0
total_error = 1

while total_error != 0:
    total_error = 0
    dw1 = dw2 = db = 0

    # Tum veri setini tek batch olarak isle
    for x1, x2, target in data:
        net = w1 * x1 + w2 * x2 + bias
        prediction = 1 if net > 0 else 0
        error = target - prediction

        total_error += abs(error)
        dw1 += learning_rate * error * x1
        dw2 += learning_rate * error * x2
        db += learning_rate * error

    if total_error != 0:
        w1 += dw1
        w2 += dw2
        bias += db

    epoch += 1

print("Egitim tamamlandi!")
print("Epoch:", epoch)
print("Toplam hata:", total_error)
print("w1:", w1, "w2:", w2, "bias:", bias)