# Cell 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# Cell 2: Definisi kelas dasar ANFIS
class ANFIS:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.consequences = np.random.randn(4, 3)  # 4 rules, 3 parameters (p,q,r) each
        self.training_history = []
        self.validation_history = []

# Cell 3: Implementasi Layer 1 - Fuzzifikasi
def layer1_membership(self, x1, x2, c11, c12, c21, c22, a11, a12, a21, a22):
    """Layer 1: Fuzzifikasi menggunakan fungsi keanggotaan bell"""
    def bell_membership(x, mean, std_dev):
        return 1 / (1 + ((x - mean) / std_dev) ** 2)
    
    µA1 = bell_membership(x1, c11, a11)
    µA2 = bell_membership(x1, c21, a21)
    µB1 = bell_membership(x2, c12, a12)
    µB2 = bell_membership(x2, c22, a22)
    
    return µA1, µA2, µB1, µB2

ANFIS.layer1_membership = layer1_membership

# Cell 4: Implementasi Layer 2-5
def layer2_firing_strength(self, µA1, µA2, µB1, µB2):
    """Layer 2: Aplikasi Operator Fuzzy (AND)"""
    w1 = µA1 * µB1
    w2 = µA2 * µB2
    return np.array([w1, w2])

def layer3_normalize(self, firing_strengths):
    """Layer 3: Normalisasi"""
    return firing_strengths / (np.sum(firing_strengths) + 1e-10)

def layer4_consequent(self, norm_firing_strengths, x1, x2):
    """Layer 4: Defuzzifikasi"""
    outputs = []
    inputs = np.array([x1, x2, 1])
    for i, w in enumerate(norm_firing_strengths):
        f = np.dot(self.consequences[i], inputs)
        outputs.append(w * f)
    return np.array(outputs)

def layer5_output(self, rule_outputs):
    """Layer 5: Output"""
    return np.sum(rule_outputs)

ANFIS.layer2_firing_strength = layer2_firing_strength
ANFIS.layer3_normalize = layer3_normalize
ANFIS.layer4_consequent = layer4_consequent
ANFIS.layer5_output = layer5_output

# Cell 5: Forward Pass
def forward_pass(self, x1, x2, mean_std_data):
    """Melakukan forward pass untuk satu set input"""
    # Ekstrak parameter
    c11, c12 = mean_std_data.loc[0, ['Mean x1', 'Mean x2']].to_numpy()
    c21, c22 = mean_std_data.loc[1, ['Mean x1', 'Mean x2']].to_numpy()
    a11, a12 = mean_std_data.loc[0, ['Std Dev x1', 'Std Dev x2']].to_numpy()
    a21, a22 = mean_std_data.loc[1, ['Std Dev x1', 'Std Dev x2']].to_numpy()
    
    # Layer 1-5
    membership = self.layer1_membership(x1, x2, c11, c12, c21, c22, a11, a12, a21, a22)
    firing_strengths = self.layer2_firing_strength(*membership)
    norm_firing_strengths = self.layer3_normalize(firing_strengths)
    rule_outputs = self.layer4_consequent(norm_firing_strengths, x1, x2)
    final_output = self.layer5_output(rule_outputs)
    
    return final_output, {
        'membership': membership,
        'firing_strengths': firing_strengths,
        'norm_firing_strengths': norm_firing_strengths,
        'rule_outputs': rule_outputs
    }

ANFIS.forward_pass = forward_pass

# Cell 6: Training Function
def train(self, X_train, y_train, X_val, y_val, mean_std_data):
    """Training ANFIS menggunakan hybrid learning"""
    print("Memulai training ANFIS...")
    
    for epoch in tqdm(range(self.epochs)):
        # Forward pass untuk semua data training
        train_predictions = []
        all_norm_firing_strengths = []
        
        for i in range(len(X_train)):
            pred, details = self.forward_pass(X_train.iloc[i, 0], X_train.iloc[i, 1], mean_std_data)
            train_predictions.append(pred)
            all_norm_firing_strengths.append(details['norm_firing_strengths'])
        
        # Update parameter konsekuen
        A = np.zeros((len(X_train), 12))
        for i in range(len(X_train)):
            norm_w = all_norm_firing_strengths[i]
            x1, x2 = X_train.iloc[i]
            row = []
            for w in norm_w:
                row.extend([w*x1, w*x2, w])
            A[i] = row
        
        try:
            optimal_params = np.linalg.lstsq(A, y_train, rcond=None)[0]
            self.consequences = optimal_params.reshape(4, 3)
        except np.linalg.LinAlgError:
            print("Warning: LSE tidak konvergen")
            continue
        
        # Evaluasi
        train_mse = mean_squared_error(y_train, train_predictions)
        val_predictions = self.predict(X_val, mean_std_data)
        val_mse = mean_squared_error(y_val, val_predictions)
        
        self.training_history.append(train_mse)
        self.validation_history.append(val_mse)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Training MSE: {train_mse:.4f}")
            print(f"Validation MSE: {val_mse:.4f}")

ANFIS.train = train

# Cell 7: Prediction Function
def predict(self, X, mean_std_data):
    """Melakukan prediksi untuk multiple input"""
    predictions = []
    for i in range(len(X)):
        pred, _ = self.forward_pass(X.iloc[i, 0], X.iloc[i, 1], mean_std_data)
        predictions.append(pred)
    return np.array(predictions)

ANFIS.predict = predict

# Cell 8: Evaluation Metrics
def mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(self, X, y_true, mean_std_data):
    """Evaluasi model menggunakan berbagai metrik"""
    y_pred = self.predict(X, mean_std_data)
    
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    
    return metrics, y_pred

ANFIS.evaluate = evaluate

# Cell 9: Visualization Functions
def plot_training_history(self):
    """Visualisasi history training"""
    plt.figure(figsize=(10, 6))
    plt.plot(self.training_history, label='Training MSE')
    plt.plot(self.validation_history, label='Validation MSE')
    plt.title('ANFIS Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_prediction_comparison(self, y_true, y_pred):
    """Visualisasi perbandingan hasil prediksi"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show()

def plot_error_distribution(self, y_true, y_pred):
    """Visualisasi distribusi error"""
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

ANFIS.plot_training_history = plot_training_history
ANFIS.plot_prediction_comparison = plot_prediction_comparison
ANFIS.plot_error_distribution = plot_error_distribution

# Cell 10: Model Saving and Loading Functions
def save_model(self, filepath):
    """Menyimpan model ANFIS"""
    with open(filepath, 'wb') as f:
        pickle.dump({
            'consequences': self.consequences,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'training_history': self.training_history,
            'validation_history': self.validation_history
        }, f)
    print(f"Model berhasil disimpan ke {filepath}")

def load_model(filepath):
    """Memuat model ANFIS yang sudah disimpan"""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    model = ANFIS(
        learning_rate=model_data['learning_rate'],
        epochs=model_data['epochs']
    )
    model.consequences = model_data['consequences']
    model.training_history = model_data['training_history']
    model.validation_history = model_data['validation_history']
    
    return model

ANFIS.save_model = save_model
ANFIS.load_model = staticmethod(load_model)

# Cell 11: Contoh Penggunaan
def example_usage():
    # Baca data
    data = pd.read_csv('data_latih.csv')
    mean_std = pd.read_csv('mean_std_hasil.csv')
    
    # Pisahkan fitur dan target
    X = data[['x1 (2 jam sebelumnya)', 'x2 (1 jam sebelumnya)']]
    y = data['target']  # Sesuaikan dengan nama kolom target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Training model
    model = ANFIS(learning_rate=0.01, epochs=100)
    model.train(X_train, y_train, X_val, y_val, mean_std)
    
    # Evaluasi
    metrics, y_pred = model.evaluate(X_test, y_test, mean_std)
    print("\nHasil Evaluasi:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualisasi
    model.plot_training_history()
    model.plot_prediction_comparison(y_test, y_pred)
    model.plot_error_distribution(y_test, y_pred)
    
    # Simpan model
    model.save_model('anfis_model.pkl')
    
    # Contoh load model dan prediksi data baru
    loaded_model = ANFIS.load_model('anfis_model.pkl')
    new_data = pd.DataFrame({
        'x1 (2 jam sebelumnya)': [10, 15, 20],
        'x2 (1 jam sebelumnya)': [12, 17, 22]
    })
    predictions = loaded_model.predict(new_data, mean_std)
    print("\nPrediksi untuk data baru:", predictions)

# Cell 12: Jalankan contoh jika diperlukan
if __name__ == "__main__":
    example_usage()