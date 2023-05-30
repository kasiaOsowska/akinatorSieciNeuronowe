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
from NET1 import predict1
from NET2 import net2
from NET2 import predict2


conn = sqlite3.connect('dishes.db')
data = pd.read_sql_query("SELECT * FROM dishes", conn)
cursor = conn.cursor()
cursor.execute('SELECT * FROM dishes')
values_to_remove = ['name', 'id']


indeks_names = [description[0] for description in cursor.description]
indeks_names = [y for y in indeks_names if y not in values_to_remove]
conn.close()
label_encoder = LabelEncoder()

data = data.drop(['id'], axis=1)
dish_names = data['name']
dish_names = np.array(dish_names).tolist()
# siec 2

label_encoder_2 = LabelEncoder()
vector_features = []
vector_dish = []

for i in range(0,17):
    vector_features.append(0.5)

dif = 3

for i in range(0,22):
    vector_dish.append(dif)



guessed = False


model_features = net1(label_encoder)
model_dish = net2(label_encoder_2)

count = 6
treshold = 0.999

while not guessed:

    a = predict1(model_features,vector_dish,label_encoder)

    print("Czy podana potrawa jest ",a)
    wybor = input("Wprowadź wartość t lub f\n")

    index = indeks_names.index(a)

    if wybor == 't':
        vector_features[index] = 5
    else:
        vector_features[index] = 0

    a,b,c = predict2(model_dish,vector_features,label_encoder_2,count)


    for name in a:
        index = dish_names.index(name)
        if vector_dish[index] >= dif:
            vector_dish[index] = vector_dish[index]-dif

    if b > treshold or count == 21:
        print("wynik to ", c)
        break

    print("a: ",a," b: ",b," c: ",c)

    count = count = count + int((22 - count) / 2)





