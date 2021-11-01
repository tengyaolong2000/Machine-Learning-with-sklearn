import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt


#import keras libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Deep-Learning-Workshop/master/Chapter02/Datasets/data.csv')


plt.scatter(df[df['label']==0]['x1'], df[df['label']==0]['x2'], marker='*')
plt.scatter(df[df['label']==1]['x1'], df[df['label']==1]['x2'], marker='<')
plt.show()

x_input = df[['x1', 'x2']].values
y_label = df['label'].values

model = Sequential()
model.add(Dense(units=50, input_dim=2, activation='relu'))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=1, input_dim=2, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_input, y_label, epochs=50)