#app/main.py

from flask import Flask

app = Flask(__name__)

@app.route("/")
def home_view():
        import requests
        import cv2
        url = 'https://nirogi.rajasthan.gov.in/assets/CM_Photo20.12.2018-813402d9d955f38b4bdda84daa89af08a47e14cd77f4c9659b8ee7260bbe5763.jpg'
        r = requests.get(url, allow_redirects=True)
        open('facebook.png', 'wb').write(r.content)
        import tensorflow as tf
        from tensorflow.keras.models import load_model       
        prediction=tf.keras.models.load_model("cnn_model_skin1.h5")
        from tensorflow.keras.preprocessing import image
        import numpy as np
        import glob
        img = cv2.imread('84.jpg')
        img = cv2.resize(img,(224,224))     # resize image to match model's expected sizing
        # img = np.reshape(img,[1,224,224,3]) # return the image with shaping that TF wants.
        cv2.imwrite("process-file.jpg",img)
        img2=image.load_img('process-file.jpg')
        x = image.img_to_array(img2)
        x = np.expand_dims(x, axis=0)
        p=np.argmax(prediction.predict(x))
        if p==0:
                a = {'value':'Benign'}
                return a
        elif p==1:
                a = {'value':'Malignant'}
                return a