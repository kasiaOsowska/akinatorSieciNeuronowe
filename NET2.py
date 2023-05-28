import sqlite3
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def net2(X_train,y_train,label_encoder,vector):
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

    predictions = model.predict([vector])
    # print(np.max(predictions))
    # print([list(X_train.iloc[0, :])])
    # print([vector])
    predicted_classes = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    return predicted_classes,np.max(predictions)