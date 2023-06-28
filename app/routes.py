import os
from flask import render_template,request, Response
from werkzeug.utils import redirect
from app import app, APP_ROOT

import io
from flask import Response
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
import keras.utils as image  

global_x = []
global_y = []
global_imgmodtime = 0

@app.route('/')
def home():
    return render_template('index.html', title='Home')

@app.route('/about')
def about():
    return render_template('about.html', title='About', name='Fernando')

@app.route('/predict')
def predict():
    global global_x, global_y, global_imgmodtime
    img_target = os.path.join(APP_ROOT, 'temp\\temp.png')
        
    dog = dog_detector(img_target)
    face = face_detector(img_target)

    if dog or face:
        global_imgmodtime = os.stat(img_target).st_mtime
        global_x, global_y = topk_predictions(img_target)
        text = 'Your predicted breeds...'

    else:
        text='Error no face or dog detected!!!'
    
    return render_template('predict.html', title='Prediction', output=text)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    target = os.path.join(APP_ROOT, 'temp\\')
    if request.method == 'POST':
        file = request.files['img']
        file.save("".join([target, 'temp.png'])) #Save file in temp folder
    
    print("upload completed")
    return render_template('predict.html', title='Complete', output='Image Uploaded!!!')

@app.route('/plot')
def plot_png():
    global global_imgmodtime
    img_path = os.path.join(APP_ROOT, 'temp\\temp.png')
    lastmodtime = os.stat(img_path).st_mtime
    print('modified time', lastmodtime)
    
    if global_imgmodtime == lastmodtime:
        plot_bar = True
    else:
        plot_bar = False
    
    fig = create_figure(img_path, plot_bar)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure(img_path, hbar):
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    display_img = image.load_img(img_path)
    ax[0].imshow(display_img)
    ax[0].axis('off')

    ax[1].axis('off')

    if hbar:
        global global_x, global_y
        plt.sca(ax[2])
        plt.title("Breed Prediction", fontsize=13)
        plt.xlabel("Percentage", fontsize=11)
        plt.yticks(range(len(global_y)), global_x)
        ax[2].barh(range(len(global_y)), global_y, color="lightblue", edgecolor='lightblue')

    else:
        ax[2].axis('off')
    
    return fig

###Breeds Predictor
import tensorflow as tf
import numpy as np

names_csv = os.path.join(APP_ROOT, "static\\dog_names.csv")
dog_names = []
with open(names_csv, 'r') as f:
    for line in f.readlines():
        name = line.strip().split('\\')
        dog_names.append(name[0])

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224, 3))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = tf.keras.preprocessing.image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

path = os.path.join(APP_ROOT,"models\\model.InceptionV3.h5")
InceptionV3_model = tf.keras.models.load_model(path)

def extract_InceptionV3(tensor):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def topk_predictions(img_path, dog_breeds=dog_names, k=5):
    # extract bottleneck features
    bottleneck_feature_InceptionV3 = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = InceptionV3_model.predict(bottleneck_feature_InceptionV3)
    
    sorted_list = [(breeds, probs) for breeds, probs in zip(dog_breeds, (predicted_vector[0]*100).round())]
    sorted_list.sort(key=lambda a: a[1], reverse=True)
    
    breeds_sorted = []
    probs_sorted = []

    for i in sorted_list[:k][::-1]:
        breeds_sorted.append(i[0])
        probs_sorted.append(i[1])
        
    return breeds_sorted, probs_sorted


#Face Detector Function
import cv2

face_path = 'app/models/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(face_path)

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

#Dog Detector
ResNet50_model = ResNet50(weights='imagenet')

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))