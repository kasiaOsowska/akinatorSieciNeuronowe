import sqlite3
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def net1(label_encoder):

    conn = sqlite3.connect('dishes.db')
    data = pd.read_sql_query("SELECT * FROM dishes", conn)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM dishes')

    values_to_remove = ['name', 'id']
    y_train = [description[0] for description in cursor.description]
    y_train = [y for y in y_train if y not in values_to_remove]

    data = data.drop(['id'], axis=1)  # usuwamy id z danych wejściowych
    output_classes = data['name'].unique()  # możliwe klasy

    cursor = conn.cursor()
    cursor.execute('SELECT * FROM dishes')

    X_train = pd.read_sql_query('SELECT * FROM dishes', conn)
    conn.close()

    X_train = X_train.transpose()

    X_train = X_train.drop(["id", "name"], axis=0)
    X_train = X_train.iloc[:, 0:].astype(int)

    num_features = X_train.shape[1]

    y_encoded = label_encoder.fit_transform(y_train)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_train = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))


    # print(len(y_train)) = 17
    # print(X_train.shape[1]) = 22
    a = int((X_train.shape[1]+len(y_train))/2)

    input_shape = (X_train.shape[1],)  # Kształt wektora cech (jest ich 17)
    model = Sequential()  # sequential to rodzaj modelu który składa się z kolejnych warstw neuronowych
    model.add(Dense(X_train.shape[1], activation='relu', input_shape=input_shape))  # warswta wejściowa zawiera 64 neurony ilosc cech
    model.add(Dense(a, activation='relu'))  # warstwa ukryta ilosc pomiedzy 1 i 3
    model.add(Dense(len(y_train), activation='softmax'))  # warstwa wyjściowa zawiera liczbe neuronów równa liczbie potraw (klas)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    history = model.fit(X_train, y_train, epochs=250, batch_size=12)  # epochs - liczba epok, batch_size - liczba próbek na jedną aktualizację wag (parametr optymalizacji algorytmu)
    return model

def predict1(model,vector,label_encoder):

    predictions = model.predict([list(vector)])
    # print("predictions")
    # print(predictions)
    # print("xtrain")
    # print(np.max(predictions))

    index = znajdz_indeks(predictions[0], 0.4)

    predicted_classes = label_encoder.inverse_transform([index])

    # middle_index = predictions.shape[1] // 2
    # # middle_index = middle_index+int(middle_index/2)
    # # print("ala")
    # # print(middle_index)
    # sorted_indices = np.argsort(predictions, axis=1)
    # middle_indices = sorted_indices[:, middle_index]
    # predicted_classes = label_encoder.inverse_transform(middle_indices)

    return (predicted_classes)

def znajdz_indeks(tablica, wartosc):
    indeks = np.abs(tablica - wartosc).argmin()
    return indeks