import pandas as pd
import numpy as np

# Langkah 1: Membaca file clustering_fcm.csv yang dihasilkan sebelumnya
data_clustering = pd.read_csv('clustering_fcm.csv')

# Langkah 2: Mengambil data yang diperlukan (x1, x2, dan derajat keanggotaan)
x1 = data_clustering['x1'].to_numpy()
x2 = data_clustering['x2'].to_numpy()
membership_cluster_1 = data_clustering['Derajat Keanggotaan Cluster 1'].to_numpy()
membership_cluster_2 = data_clustering['Derajat Keanggotaan Cluster 2'].to_numpy()

# Fungsi untuk menghitung weighted mean
def weighted_mean(values, memberships):
    return np.sum(memberships * values) / np.sum(memberships)

# Fungsi untuk menghitung weighted standar deviasi
def weighted_std(values, memberships, mean):
    return np.sqrt(np.sum(memberships * (values - mean) ** 2) / np.sum(memberships))

# Menghitung mean dan standar deviasi untuk Cluster 1
mean_x1_cluster_1 = np.round (weighted_mean(x1, membership_cluster_1), 5)
mean_x2_cluster_1 = np.round (weighted_mean(x2, membership_cluster_1), 5)
std_x1_cluster_1 = np.round (weighted_std(x1, membership_cluster_1, mean_x1_cluster_1), 5)
std_x2_cluster_1 = np.round (weighted_std(x2, membership_cluster_1, mean_x2_cluster_1), 5)

# Menghitung mean dan standar deviasi untuk Cluster 2
mean_x1_cluster_2 = np.round (weighted_mean(x1, membership_cluster_2), 5)
mean_x2_cluster_2 = np.round (weighted_mean(x2, membership_cluster_2), 5)
std_x1_cluster_2 = np.round (weighted_std(x1, membership_cluster_2, mean_x1_cluster_2), 5)
std_x2_cluster_2 = np.round (weighted_std(x2, membership_cluster_2, mean_x2_cluster_2), 5)

# Membuat tabel hasil dalam DataFrame
df_results = pd.DataFrame({
    'Cluster': ['Cluster 1', 'Cluster 2'],
    'Mean x1': [mean_x1_cluster_1, mean_x1_cluster_2],
    'Std Dev x1': [std_x1_cluster_1, std_x1_cluster_2],
    'Mean x2': [mean_x2_cluster_1, mean_x2_cluster_2],
    'Std Dev x2': [std_x2_cluster_1, std_x2_cluster_2]
})

# Simpan tabel hasil ke file CSV jika diperlukan
df_results.to_csv('mean_std_hasil.csv', index=False)

print("\nHasil mean dan standar deviasi disimpan ke 'mean_std_hasil.csv'")
