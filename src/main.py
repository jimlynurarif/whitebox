import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from FFNN import FFNN
import pickle

def get_layer_sizes():
    while True:
        try:
            layer_input = input("Masukkan ukuran layer (pisahkan dengan koma, contoh: 784,128,64,10): ")
            layer_sizes = list(map(int, layer_input.split(',')))
            if len(layer_sizes) < 2:
                print("Minimal dua layer (input dan output).")
                continue
            return layer_sizes
        except ValueError:
            print("Input tidak valid. Masukkan bilangan bulat yang dipisahkan koma.")

def get_activations(num_layers):
    activations = []
    activation_options = ['linear', 'relu', 'sigmoid', 'tanh', 'softmax']
    for i in range(num_layers):
        while True:
            activation = input(f"Pilih aktivasi untuk layer {i+1} ({', '.join(activation_options)}): ").lower()
            if activation in activation_options:
                activations.append(activation)
                break
            else:
                print("Fungsi aktivasi tidak valid.")
    return activations

def get_loss_function():
    loss_options = ['mse', 'binary_cross_entropy', 'categorical_cross_entropy']
    while True:
        loss = input(f"Pilih fungsi loss ({', '.join(loss_options)}): ").lower()
        if loss in loss_options:
            return loss
        else:
            print("Fungsi loss tidak valid.")

def get_weight_init_params(method):
    params = {'method': method}
    if method == 'uniform':
        lower = float(input("Masukkan batas bawah distribusi uniform: "))
        upper = float(input("Masukkan batas atas distribusi uniform: "))
        params['lower'] = lower
        params['upper'] = upper
    elif method == 'normal':
        mean = float(input("Masukkan mean distribusi normal: "))
        var = float(input("Masukkan varian distribusi normal: "))
        params['mean'] = mean
        params['variance'] = var
    return params

def get_weight_inits(num_layers):
    weight_inits = []
    for i in range(num_layers):
        print(f"\nInisialisasi bobot untuk layer {i+1}:")
        while True:
            method = input("Pilih metode (zero, uniform, normal): ").lower()
            if method in ['zero', 'uniform', 'normal']:
                params = get_weight_init_params(method)
                weight_inits.append(params)
                break
            else:
                print("Metode tidak valid.")
    return weight_inits

def main():
    pilihan = input("Muat model (L) atau latih model baru (T)? ").upper()
    
    if pilihan == 'L':
        path_model = input("Masukkan path model: ")
        path_data = input("Masukkan path data: ")
        
        # Muat model
        model = FFNN.load_model(path_model)
        
        # Muat data
        data = np.load(path_data)
        X = data['X']
        y = data['y']
        
        # Preproses data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # One-hot encoding jika diperlukan
        if model.loss_function == 'categorical_cross_entropy':
            encoder = OneHotEncoder(sparse_output=False)
            y_onehot = encoder.fit_transform(y.reshape(-1, 1))
        else:
            y_onehot = y
        
        # Evaluasi
        prediksi = model.forward(X_scaled)
        loss = model._compute_loss(prediksi, y_onehot)
        
        if model.loss_function == 'categorical_cross_entropy':
            pred_kelas = np.argmax(prediksi, axis=1)
            akurasi = np.mean(pred_kelas == y.astype(int))
            print(f"Loss: {loss:.4f}, Akurasi: {akurasi*100:.2f}%")
        else:
            print(f"Loss: {loss:.4f}")
        
    elif pilihan == 'T':
        # Input parameter model
        layer_sizes = get_layer_sizes()
        num_layers = len(layer_sizes) - 1
        activations = get_activations(num_layers)
        loss_function = get_loss_function()
        weight_inits = get_weight_inits(num_layers)
        
        # Parameter pelatihan
        epochs = int(input("Masukkan jumlah epoch: "))
        batch_size = int(input("Masukkan ukuran batch: "))
        learning_rate = float(input("Masukkan learning rate: "))
        verbose = int(input("Verbose (0 atau 1): "))
        
        # Muat dan preproses data MNIST
        print("\nMemuat data MNIST...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X = X.astype(np.float32)
        y = y.astype(int)
        
        # Bagi data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=5000, test_size=10000, random_state=42
        )
        
        # Normalisasi
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # One-hot encoding
        if loss_function == 'categorical_cross_entropy':
            encoder = OneHotEncoder(sparse_output=False)
            y_train = encoder.fit_transform(y_train.reshape(-1, 1))
            y_test = encoder.transform(y_test.reshape(-1, 1))
        
        # Inisialisasi model
        model = FFNN(
            layer_sizes=layer_sizes,
            activations=activations,
            loss_function=loss_function,
            weight_inits=weight_inits
        )
        
        # Pelatihan
        print("\nMelatih model...")
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose
        )
        
        # Evaluasi
        pred_test = model.forward(X_test)
        loss_test = model._compute_loss(pred_test, y_test)
        
        if loss_function == 'categorical_cross_entropy':
            pred_kelas = np.argmax(pred_test, axis=1)
            akurasi = np.mean(pred_kelas == y_test.astype(int))
            print(f"\nLoss: {loss_test:.4f}, Akurasi: {akurasi*100:.2f}%")
        else:
            print(f"\nLoss: {loss_test:.4f}")
        
        # Simpan model
        simpan = input("Simpan model? (y/n): ").lower()
        if simpan == 'y':
            path_simpan = input("Masukkan nama file untuk disimpan: ")
            model.save_model(path_simpan)
            print(f"Model disimpan di {path_simpan}")
        
    else:
        print("Pilihan tidak valid.")

if __name__ == "__main__":
    main()