import sqlite3
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def net2(label_encoder):

    conn = sqlite3.connect('dishes.db')
    data = pd.read_sql_query("SELECT * FROM dishes", conn)

    data = data.drop(['id'], axis=1)  # usuwamy id z danych wejściowych
    output_classes = data['name'].unique()  # możliwe klasy
    y_train = data['name']
    names = y_train
    X_train = data.drop(['name'], axis=1)  # usuwamy kolumnę 'name' z danych wejściowych

    y_encoded_2 = label_encoder.fit_transform(y_train)

    onehot_encoder_2 = OneHotEncoder(sparse_output=False)
    y_train = onehot_encoder_2.fit_transform(y_encoded_2.reshape(-1, 1))

    # print(len(y_train)) = 22
    # print(X_train.shape[1]) = 17
    a = int((X_train.shape[1] + len(y_train)) / 2)

    input_shape = (X_train.shape[1],)  # Kształt wektora cech (jest ich 17)
    model = Sequential()  # sequential to rodzaj modelu który składa się z kolejnych warstw neuronowych
    model.add(Dense(X_train.shape[1], activation='relu', input_shape=input_shape))  # warswta wejściowa zawiera 64 neurony ilosc cech
    model.add(Dense(a, activation='relu'))  # warstwa ukryta ilosc pomiedzy 1 i 3
    model.add(Dense(len(y_train), activation='softmax'))  # warstwa wyjściowa zawiera liczbe neuronów równa liczbie potraw (klas)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    history = model.fit(X_train, y_train, epochs=250, batch_size=12)  # epochs - liczba epok, batch_size - liczba próbek na jedną aktualizację wag (parametr optymalizacji algorytmu)

    return model

def predict2(model,vector,label_encoder,count):

    predictions = model.predict([vector])

    najmniejsze_indeksy = np.argsort(predictions, axis=1)[:, :count]
    predicted_classes = label_encoder.inverse_transform(najmniejsze_indeksy.flatten())

    predicted_win = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    return predicted_classes,np.max(predictions),predicted_win