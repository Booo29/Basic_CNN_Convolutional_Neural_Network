import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

model = tf.keras.models.load_model('model2.keras')
print("Loaded model.")

(_, _), (x_test, y_test) = datasets.cifar10.load_data()
x_test = x_test / 255.0

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def Load_Images(model, x_test, y_test):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i])
        prediccion = np.argmax(model.predict(x_test[i:i+1]), axis=1)[0]
        etiqueta_real = y_test[i][0]
        plt.xlabel(f"Pred: {class_names[prediccion]}, Real: {class_names[etiqueta_real]}")
    plt.show()

Load_Images(model, x_test, y_test)
