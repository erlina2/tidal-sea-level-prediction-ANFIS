import pandas as pd #Digunakan untuk membaca dan menyimpan data

# Baca file Excel yang berisi data pasang surut
identifikasi_variabel=pd.read_excel('data_mentah.xlsx')

# Hilangkan kolom pertama (yang berisi tanggal)
identifikasi_variabel = identifikasi_variabel.drop(columns=[identifikasi_variabel.columns[0]])

# Inisialisasi list untuk menyimpan hasil pengelompokan
rows = []

# Mulai dari baris pertama dan iterasi melalui kolom (jam)
for date in identifikasi_variabel.index:  # Iterasi melalui setiap baris (tanggal)
    for hour in range(2, identifikasi_variabel.shape[1]):  # Iterasi mulai dari kolom ke-3 (jam ke-3 dan seterusnya)
        x1 = identifikasi_variabel.iloc[date, hour - 2]  # x1: data pasang surut 2 jam sebelumnya
        x2 = identifikasi_variabel.iloc[date, hour - 1]  # x2: data pasang surut 1 jam sebelumnya
        y = identifikasi_variabel.iloc[date, hour]       # y: data pasang surut pada jam saat ini
        rows.append([x1, x2, y])  # Simpan hasil dalam list

# Ubah list menjadi DataFrame untuk hasil akhirnya
result_identifikasi_variabel= pd.DataFrame(rows, columns=['x1 (2 jam sebelumnya)', 'x2 (1 jam sebelumnya)', 'y (pasang surut saat ini)'])

# Tampilkan hasil
#print(result_df.head(10))  # Tampilkan 10 hasil pertama agar sesuai dengan contoh

# Simpan hasil dalam file CSV
result_identifikasi_variabel.to_csv('data_input.csv', index=False)
