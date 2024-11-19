import pandas as pd # #Digunakan untuk membaca dan menyimpan data
import numpy as np # Digunakan untuk operasi matematis

# Membaca file csv untuk data latih
data_input = pd.read_csv('data_input.csv')

# Ubah data menjadi array numpy dengan tipe data float
data = data_input.to_numpy(dtype=float)

# Memisahkan fitur (x1 dan x2) dan target (y)
X = data[:, :-1]  # Semua kolom kecuali yang terakhir (x1 dan x2)
Y = data[:, -1]   # Kolom terakhir adalah target (y)

# 1. Penetapan parameter FCM
c = 2  # Banyak cluster
w = 2.0  # Pangkat fuzziness
MaxIter = 100  # Maksimum Iterasi
epsilon = 0.01  # Toleransi kesalahan
P0 = 0  # Fungsi objektif awal
t = 1  # Iterasi awal

# 2. Penetapan Nilai Matriks Keanggotaan (U) secara acak
U = np.random.rand(X.shape[0], c)
U = U / U.sum(axis=1, keepdims=True)  # Normalisasi agar setiap baris berjumlah 1

# 3. Hitung pusat cluster atau centroid
def pusat_cluster(U, data, w):
    pc = U ** w  # Kuadratkan nilai keanggotaan
    return (pc.T @ data) / pc.sum(axis=0)[:, None]

# 4. Hitung fungsi objectif
def fungsi_objectif(U, centers, data, w):
    objektif = 0
    for k in range(c): # Iterasi untuk setiap cluster
        for i in range(len(data)): # Iterasi untuk setiap data
            jarak_ke_centroid = np.linalg.norm(data[i] - centers[k]) ** 2 # Hitung jarak ke centroid 
            objektif += (U[i, k] ** w) * jarak_ke_centroid # Tambahkan jarak berbobot ke fungsi objektif
    return objektif

# 5. Perbarui nilai matriks keanggotaan (U)
def update_membership(U, centers, data, w):
    U_new = np.zeros(U.shape)
    for i in range(len(data)):
        for k in range(c):
            numerator = np.linalg.norm(data[i] - centers[k]) ** 2
            sum_term = 0
            for j in range(c):
                denominator = np.linalg.norm(data[i] - centers[j]) ** 2
                sum_term += (numerator / denominator) ** (1 / (w - 1))
            U_new[i, k] = 1 / sum_term

    # **Normalisasi agar setiap baris matriks U berjumlah 1**
    U_new = U_new / U_new.sum(axis=1, keepdims=True)
    return U_new

# 3. Menyimpan nilai fungsi objektif sebelumnya
previous_objective_value = None

# Langkah 6: Algoritma FCM dan cek kondisi berhenti
for t in range(1, MaxIter + 1):
    # Menghitung centroid cluster (v_ij)
    centers = pusat_cluster(U, X, w)
    
    # Menghitung fungsi objektif (P_t)
    objective_value = fungsi_objectif(U, centers, X, w)
    
    # Jika ini bukan iterasi pertama, hitung perbedaan fungsi objektif
    if t==1:
        print(f"Iterasi ke-{t}, Fungsi Objektif: {objective_value:.5f} (Iterasi Pertama)")
    else:
        #Untuk iterasi setelah yang pertama, bandingkan dengan fungsi objektif sebelumnya
        objective_difference = abs(objective_value - previous_objective_value)
        print(f"Iterasi ke-{t}, Fungsi Objektif: {objective_value:.5f}, Selisih: {objective_difference:.5f}")
        
        # Jika perbedaan fungsi objektif kurang dari epsilon, maka berhenti
        if objective_difference < epsilon:
            print(f"Iterasi berhenti karena selisih fungsi objektif lebih kecil dari epsilon ({epsilon:.5f}) pada iterasi ke-{t}")
            break
    
    # Simpan fungsi objektif saat ini untuk iterasi berikutnya
    previous_objective_value = objective_value
    
    # Perbarui matriks keanggotaan (μ_ik)
    U_new = update_membership(U, centers, X, w)
    
    # Hitung perubahan keanggotaan
    if np.linalg.norm(U_new - U) < epsilon:
        print(f"Iterasi berhenti karena perubahan keanggotaan lebih kecil dari epsilon pada iterasi ke-{t}")
        break
    U = U_new

# Setelah iterasi selesai, hitung nilai fungsi objektif akhir
objective_value = fungsi_objectif(U, centers, X, w)

# Cetak hasil ketika iterasi berhenti
#print(f"Fungsi objektif pada iterasi terakhir: {objective_value:.5f}")
print("Pusat cluster pada iterasi terakhir:")
print(np.round(centers, 5)) # Batasi 5 angka di belakang koma untuk pusat cluster
print("Matriks keanggotaan pada iterasi terakhir:")
print(np.round(U, 5))  # Batasi 5 angka di belakang koma untuk matriks keanggotaan

# 7. Membuat tabel kecenderungan masuk cluster
data_list = []

for i in range(len(X)):
    cluster_1 = round(U[i, 0], 5)
    cluster_2 = round(U[i, 1], 5)
    tendency = "C1" if cluster_1 > cluster_2 else "C2"  # Kecenderungan masuk cluster
    data_list.append([X[i, 0], X[i, 1], cluster_1, cluster_2, tendency])

# Konversi list ke DataFrame
df_result = pd.DataFrame(data_list, columns=[ 'x1', 'x2', 'Derajat Keanggotaan Cluster 1', 'Derajat Keanggotaan Cluster 2', 'Kecenderungan Masuk Cluster'])

# Simpan DataFrame ke file CSV
df_result.to_csv('clustering_fcm.csv', index=False)

print("\nHasil clustering disimpan ke 'clustering_fcm.csv'")
