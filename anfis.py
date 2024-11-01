import pandas as pd
import numpy as np

# Langkah 1: Membaca file CSV untuk data latih dan data mean & standar deviasi
data_latih = pd.read_csv('data_latih.csv')
mean_std = pd.read_csv('mean_std_hasil.csv')

# Ambil nilai x1 dan x2 dari data_latih.csv
x1_values = data_latih['x1 (2 jam sebelumnya)'].to_numpy()
x2_values = data_latih['x2 (1 jam sebelumnya)'].to_numpy()

# Ambil mean dan standar deviasi dari mean_std_hasil.csv
c11, c12 = mean_std.loc[0, ['Mean x1', 'Mean x2']].to_numpy()  # Mean untuk Cluster 1
c21, c22 = mean_std.loc[1, ['Mean x1', 'Mean x2']].to_numpy()  # Mean untuk Cluster 2
a11, a12 = mean_std.loc[0, ['Std Dev x1', 'Std Dev x2']].to_numpy()  # Std Dev untuk Cluster 1
a21, a22 = mean_std.loc[1, ['Std Dev x1', 'Std Dev x2']].to_numpy()  # Std Dev untuk Cluster 2

# Fungsi untuk menghitung nilai keanggotaan (µ) menggunakan rumus yang diberikan
def derajat_keanggotaan(x, mean, std_dev):
    return 1 / (1 + ((x - mean) / std_dev) ** 2)

# Inisialisasi list untuk menyimpan hasil
results = []

# Hitung nilai µA1, µA2, µB1, dan µB2 untuk setiap data x1 dan x2
for idx in range(len(x1_values)):
    x1 = x1_values[idx]
    x2 = x2_values[idx]

    µA1 = round(derajat_keanggotaan(x1, c11, a11), 5)
    µA2 = round(derajat_keanggotaan(x1, c21, a21), 5)
    µB1 = round(derajat_keanggotaan(x2, c12, a12), 5)
    µB2 = round(derajat_keanggotaan(x2, c22, a22), 5)

    # Simpan hasil dengan format yang mirip tabel
    results.append([µA1, µA2, µB1, µB2 ])

# Konversi hasil ke DataFrame
df_results = pd.DataFrame(results, columns=['µA1', 'µA2', 'µB1', 'µB2'])

# Tampilkan sebagian hasil untuk memastikan
print(df_results.head(10))

# Simpan hasil ke file CSV
df_results.to_csv('anfis_layer1_results.csv', index=False)
print("\nHasil ANFIS Layer 1 disimpan ke 'anfis_layer1_results.csv'")
