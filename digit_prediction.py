import gradio as gr
import tensorflow as tf
import numpy as np
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
print(x_train[0])

import matplotlib.pyplot as plt
plt.imshow(x_train[0],cmap='gray')
plt.show()

x_train=x_train/255.0
x_test=x_test/255.0
x_train[0]

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)
y_train[0]

y_train=tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test=tf.keras.utils.to_categorical(y_test,num_classes=10)
y_train[0]

model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
input_shape=(28,28,1)
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

model.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

def sketch_recognition(img):
    x=model.predict(img.reshape(1,28,28)).argmax(axis=1)
    return x

interface=gr.Interface(fn=sketch_recognition,inputs="sketchpad",outputs='text')
interface.launch(debug=True)