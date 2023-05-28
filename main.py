import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

import sqlite3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from data import DataHandler, Dish
from compareVectors import getBest
from entropy import getBestFeature, getBestFeatureGini
from treeToPng import treeToPng
from NET1 import net1
from NET2 import net2


conn = sqlite3.connect('dishes.db')
data = pd.read_sql_query("SELECT * FROM dishes", conn)
cursor = conn.cursor()
cursor.execute('SELECT * FROM dishes')

values_to_remove = ['name', 'id']
y_train = [description[0] for description in cursor.description]
y_train = [y for y in y_train if y not in values_to_remove]


indeks_names = y_train
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
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)
onehot_encoder = OneHotEncoder(sparse_output=False)
y_train = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

# siec 2 aaaaaaaaaaaaaaaaaaa
conn = sqlite3.connect('dishes.db')
data = pd.read_sql_query("SELECT * FROM dishes", conn)

data = data.drop(['id'], axis=1)  # usuwamy id z danych wejściowych
output_classes = data['name'].unique()  # możliwe klasy
y_train_2 = data['name']
names = y_train_2
X_train_2 = data.drop(['name'], axis=1)  # usuwamy kolumnę 'name' z danych wejściowych

label_encoder_2 = LabelEncoder()
y_encoded_2 = label_encoder_2.fit_transform(y_train_2)

onehot_encoder_2 = OneHotEncoder(sparse_output=False)
y_train_2 = onehot_encoder_2.fit_transform(y_encoded_2.reshape(-1, 1))
vector = []

for i in range(0,17):
    vector.append(0.5)


def ask_question(id_of_feature, user_v, feature_names_df):
    choice = input()

    if choice == "f":
        user_v[id_of_feature] = False
        return "f"
    elif choice == "t":
        user_v[id_of_feature] = True
        return "t"
    else:
        print(" IDK ")
        return "n"


dh = DataHandler()
data = dh.getDishesAndNamesVector()
feature_names = Dish.get_dish_variable_names()
feature_names_df = pd.DataFrame(feature_names)
answered = False
guessed = False
depth = 0

size_of_vector = len(feature_names)
user_v = pd.DataFrame([[None] * size_of_vector], columns=range(size_of_vector))

while not guessed:

    a = net1(X_train,y_train,label_encoder)

    X_train = X_train.drop(X_train[X_train.index == a[0]].index)
    y_train = label_encoder.inverse_transform(np.argmax(y_train, axis=1))
    index = np.where(y_train == a)
    y_train = np.delete(y_train, index)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_train = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

    index = indeks_names.index(a)

    y_train_2 = label_encoder_2.inverse_transform(np.argmax(y_train_2, axis=1))
    print(y_train_2)

    print("Czy podana potrawa jest ",a)
    wybor = input("Wprowadź wartość t lub f\n")

    if wybor == 't':
        vector[index] = 1
        #X_train_2 = X_train_2.drop(X_train_2[X_train_2[a[0]] == 0].index)
        indeks = 0
        for index, row in X_train_2.iterrows():
            indeks = indeks+1
            if row[a[0]] == 0:
                print(index)
                X_train_2 = X_train_2.drop(index)
                indeks = indeks-1
                y_train_2 = np.delete(y_train_2, indeks)
    else:
        vector[index] = 0
        #X_train_2 = X_train_2.drop(X_train_2[X_train_2[a[0]] == 1].index)
        indeks = 0
        for index, row in X_train_2.iterrows():
            indeks = indeks + 1
            if row[a[0]] == 1:
                X_train_2 = X_train_2.drop(index)
                indeks = indeks - 1
                y_train_2 = np.delete(y_train_2, indeks)

    print(y_train_2)
    label_encoder_2 = LabelEncoder()
    y_encoded_2 = label_encoder_2.fit_transform(y_train_2)
    onehot_encoder_2 = OneHotEncoder(sparse_output=False)
    y_train_2 = onehot_encoder_2.fit_transform(y_encoded_2.reshape(-1, 1))


    a,b = net2(X_train_2, y_train_2, label_encoder_2,vector)

    if b>0.75:
        print("wynik to ",a)
        break




