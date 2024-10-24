import tkinter as tk
from tkinter import filedialog, Label
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = tf.keras.models.load_model('model2.keras')

def predict_image():
    if not hasattr(predict_image, "img_array"):
        messagebox.showwarning("Advertencia", "Por favor, carga una imagen primero.")
        return

    img_resized = tf.image.resize(predict_image.img_array, (32, 32)) / 255.0
    img_expanded = np.expand_dims(img_resized, axis=0)  # Añadir dimensión para el batch

    prediction = model.predict(img_expanded)
    class_index = np.argmax(prediction[0])
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predicted_class = classes[class_index]

    result_label.config(text=f"Predicción: {predicted_class}")

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = load_img(file_path, target_size=(32, 32))
        predict_image.img_array = img_to_array(img)  
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img  

def clear_image():
    image_label.config(image='')
    image_label.image = None
    result_label.config(text="Predicción: ")

root = tk.Tk()
root.title("Predicción de CIFAR-10")

image_label = Label(root)
image_label.pack(pady=20)

load_button = tk.Button(root, text="Cargar Imagen", command=load_image, bg='green', fg='white')
load_button.pack(side=tk.LEFT, padx=(20, 10), pady=(0, 20))

predict_button = tk.Button(root, text="Predecir", command=predict_image, bg='blue', fg='white')
predict_button.pack(side=tk.LEFT, padx=10, pady=(0, 20))

clear_button = tk.Button(root, text="Limpiar", command=clear_image, bg='red', fg='white')
clear_button.pack(side=tk.LEFT, padx=(10, 20), pady=(0, 20))

result_label = Label(root, text="Predicción: ")
result_label.pack(pady=(10, 20))

root.mainloop()
