import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')

# Define class names
CLASS_NAMES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

def predict_image():
    global img
    file_path = filedialog.askopenfilename()
    img = Image.open(file_path).resize((150, 150))
    img_tk = ImageTk.PhotoImage(img)
    lbl_img.config(image=img_tk)
    lbl_img.image = img_tk

    img_array = np.array(img).reshape((1, 150, 150, 3)) / 255.0
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    label_pred.configure(text="This is an image of a "+predicted_class) # Display prediction

root = tk.Tk()
root.title("Animal Classifier ML")

root.geometry("800x600")

# Style the upload button
button_upload = tk.Button(root, text="Upload an image", command=predict_image,
                          font=('Arial', 10, 'bold'), bg='#87ceeb', fg='white', height=2, width=15)
button_upload.pack(pady=20)

# Add a label for the image
lbl_img = tk.Label(root)
lbl_img.pack(pady=20)

# Add prediction label
label_pred = tk.Label(root, font=('Arial', 12, 'bold'))
label_pred.pack(pady=20)

root.mainloop()
