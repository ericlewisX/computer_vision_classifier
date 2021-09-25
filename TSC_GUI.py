## Standard
import numpy as np
import pandas as pd
import sys
sys.setrecursionlimit(1500)

## IGUI
import tkinter as tk

from tkinter import *, filedialog
from PIL import ImageTk, Image

## Tensoflow and Keras
import tensorflow as tf

# Graphical User-Interface
# load the model
model = tf.keras.models.load_model('trafficSignModel.h5', 
                                   custom_objects = None, 
                                   compile = True)

classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',
            3:'Speed limit (50km/h)',
            4:'Speed limit (60km/h)',
            5:'Speed limit (70km/h)',
            6:'Speed limit (80km/h)',
            7:'End of speed limit (80km/h)',
            8:'Speed limit (100km/h)',
            9:'Speed limit (120km/h)',
            10:'No passing',
            11:'No passing for vehicles over 3.5 metric tons',
            12:'Right-of-way at the next intersection',
            13:'Priority road',
            14:'Yield',
            15:'Stop',
            16:'No vehicles',
            17:'Vehicles over 3.5 metric tons prohibited',
            18:'No entry',
            19:'General caution',
            20:'Dangerous curve left',
            21:'Dangerous curve right',
            22:'Double curve',
            23:'Bumpy road',
            24:'Slippery road',
            25:'Road narrows on the right',
            26:'Road work',
            27:'Traffic signals',
            28:'Pedestrians',
            29:'Children crossing',
            30:'Bicycles crossing',
            31:'Beware of ice/snow',
            32:'Wild animals crossing',
            33:'End of all speed and passing limits',
            34:'Turn right ahead',
            35:'Turn left ahead',
            36:'Ahead only',
            37:'Go straight or right',
            38:'Go straight or left',
            39:'Keep right',
            40:'Keep left',
            41:'Roundabout mandatory',
            42:'End of no passing',
            43:'End of no passing by vehicles over 3.5 metric tons' }

top = tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background = '#7BB661')
label = Label(top,background = '#7BB661', font = ('arial',15,'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = resize(file_path)
    image_to_predict = process(image)
    pred = model.predict_classes([image_to_predict])[0]
    sign = classes[pred+1]
    print(sign)
    label.configure(foreground='#011638', text=sign)
    
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
    
def upload_image():
    
    file_path=filedialog.askopenfilename()
    uploaded=Image.open(file_path)
    uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
    im=ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image=im
    label.configure(text='')
    show_classify_button(file_path)

    
upload = Button(top, text="Upload an image", command = upload_image, padx = 10, pady = 5)

upload.configure(background = '#364156', foreground = 'white', font = ('arial', 10, 'bold'))

upload.pack(side = BOTTOM, pady = 50)

sign_image.pack(side = BOTTOM, expand = True)

label.pack(side = BOTTOM, expand = True)

heading = Label(top, text = "Know Your Traffic Sign", pady = 20, font = ('arial',20,'bold'))

heading.configure(background = '#7BB661', foreground = '#364156')

heading.pack()

top.mainloop()