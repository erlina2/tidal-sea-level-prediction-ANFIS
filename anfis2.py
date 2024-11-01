import pandas as pd
import numpy as np

class ANFIS:
    def __init__(self):
        self.rules = []
        self.consequences = []
    
    def layer1_membership(self, x1, x2, c11, c12, c21, c22, a11, a12, a21, a22):
        """
        Layer 1: Fuzzifikasi
        Menghitung derajat keanggotaan untuk setiap input menggunakan fungsi bell
        """
        def bell_membership(x, mean, std_dev):
            return 1 / (1 + ((x - mean) / std_dev) ** 2)
        
        µA1 = bell_membership(x1, c11, a11)
        µA2 = bell_membership(x1, c21, a21)
        µB1 = bell_membership(x2, c12, a12)
        µB2 = bell_membership(x2, c22, a22)
        
        return µA1, µA2, µB1, µB2
    
    def layer2_firing_strength(self, µA1, µA2, µB1, µB2):
        """
        Layer 2: Aplikasi Operator Fuzzy (AND)
        Menghitung firing strength untuk setiap aturan
        """
        w1 = µA1 * µB1  # Rule 1: IF x1 is A1 AND x2 is B1 THEN ...
        w2 = µA1 * µB2  # Rule 2: IF x1 is A1 AND x2 is B2 THEN ...
        w3 = µA2 * µB1  # Rule 3: IF x1 is A2 AND x2 is B1 THEN ...
        w4 = µA2 * µB2  # Rule 4: IF x1 is A2 AND x2 is B2 THEN ...
        
        return np.array([w1, w2, w3, w4])
    
    def layer3_normalize(self, firing_strengths):
        """
        Layer 3: Normalisasi
        Menghitung normalized firing strength
        """
        return firing_strengths / (np.sum(firing_strengths) + 1e-10)
    
    def layer4_consequent(self, norm_firing_strengths, x1, x2, consequences):
        """
        Layer 4: Defuzzifikasi
        Menghitung output untuk setiap aturan
        consequences: array of [p, q, r] untuk setiap aturan
        """
        outputs = []
        for i, w in enumerate(norm_firing_strengths):
            p, q, r = consequences[i]
            f = p*x1 + q*x2 + r
            outputs.append(w * f)
        return np.array(outputs)
    
    def layer5_output(self, rule_outputs):
        """
        Layer 5: Output
        Menghitung output akhir (weighted average)
        """
        return np.sum(rule_outputs)
    
    def predict(self, x1, x2, mean_std_data, consequence_params):
        """
        Melakukan prediksi untuk satu set input
        """
        # Ekstrak parameter dari data mean dan standar deviasi
        c11, c12 = mean_std_data.loc[0, ['Mean x1', 'Mean x2']].to_numpy()
        c21, c22 = mean_std_data.loc[1, ['Mean x1', 'Mean x2']].to_numpy()
        a11, a12 = mean_std_data.loc[0, ['Std Dev x1', 'Std Dev x2']].to_numpy()
        a21, a22 = mean_std_data.loc[1, ['Std Dev x1', 'Std Dev x2']].to_numpy()
        
        # Layer 1
        µA1, µA2, µB1, µB2 = self.layer1_membership(x1, x2, c11, c12, c21, c22, a11, a12, a21, a22)
        
        # Layer 2
        firing_strengths = self.layer2_firing_strength(µA1, µA2, µB1, µB2)
        
        # Layer 3
        norm_firing_strengths = self.layer3_normalize(firing_strengths)
        
        # Layer 4
        rule_outputs = self.layer4_consequent(norm_firing_strengths, x1, x2, consequence_params)
        
        # Layer 5
        final_output = self.layer5_output(rule_outputs)
        
        return final_output, {
            'membership': (µA1, µA2, µB1, µB2),
            'firing_strengths': firing_strengths,
            'norm_firing_strengths': norm_firing_strengths,
            'rule_outputs': rule_outputs
        }

# Penggunaan model
def main():
    # Baca data
    data_latih = pd.read_csv('data_latih.csv')
    mean_std = pd.read_csv('mean_std_hasil.csv')
    
    # Contoh parameter konsekuen untuk setiap aturan (should be learned/optimized)
    consequence_params = [
        [0.5, 0.5, 0],  # Rule 1: f1 = p1*x1 + q1*x2 + r1
        [0.3, 0.7, 0],  # Rule 2: f2 = p2*x1 + q2*x2 + r2
        [0.7, 0.3, 0],  # Rule 3: f3 = p3*x1 + q3*x2 + r3
        [0.4, 0.6, 0]   # Rule 4: f4 = p4*x1 + q4*x2 + r4
    ]
    
    # Inisialisasi ANFIS
    anfis = ANFIS()
    
    # List untuk menyimpan semua hasil
    all_results = []
    
    # Proses setiap baris data
    for idx, row in data_latih.iterrows():
        x1 = row['x1 (2 jam sebelumnya)']
        x2 = row['x2 (1 jam sebelumnya)']
        
        # Lakukan prediksi
        output, details = anfis.predict(x1, x2, mean_std, consequence_params)
        
        # Simpan hasil
        result = {
            'x1': x1,
            'x2': x2,
            'µA1': details['membership'][0],
            'µA2': details['membership'][1],
            'µB1': details['membership'][2],
            'µB2': details['membership'][3],
            'output': output
        }
        all_results.append(result)
    
    # Konversi ke DataFrame dan simpan
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('anfis_complete_results.csv', index=False)
    print("\nHasil ANFIS lengkap disimpan ke 'anfis_complete_results.csv'")
    
    return results_df

if __name__ == "__main__":
    results = main()
    print("\nContoh hasil 5 baris pertama:")
    print(results.head())