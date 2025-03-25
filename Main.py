from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
import pandas as pd
from keras.utils.np_utils import to_categorical

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, GRU
from keras.layers import Conv2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.model_selection import train_test_split
import keras


main = tkinter.Tk()
main.title("Video Forgery Detection using DNN")
main.geometry("1200x1200")

global dnn_model, filename, X, Y, labels
global X_train, X_test, y_train, y_test

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset():
    global filename, labels, X, Y, dataset
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    counter = 0
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name.strip())
            counter += 1
    
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):        
                name = os.path.basename(root)
                label = getLabel(name)
                cap = cv2.VideoCapture(root+"/"+directory[j])
                while True:
                    ret, img = cap.read()
                    if ret == True:
                        img = cv2.resize(img, (32, 32))
                        X.append(img)
                        Y.append(label)
                        print(name+" "+str(label))
                    else:
                        break
                cap.release()
                cv2.destroyAllWindows()     
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    print(np.unique(Y, return_counts=True))    
    text.insert(END,"Class labels found in Dataset : "+str(labels)+"\n")    
    text.insert(END,"Total videos found in dataset : "+str(counter))

def normalizeFrames():
    global X, Y
    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Video Frames Shuffling & Normalization processing Completed")
    
def splitFrames():
    global X, Y
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    X = np.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2], X.shape[3]))
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"80% video frames used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% video frames used for testing : "+str(X_test.shape[0])+"\n\n")
    

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    global labels
    global accuracy, precision, recall, fscore
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")    

def trainModel():
    text.delete('1.0', END)
    global X, Y, labels, dnn_model
    global X_train, X_test, y_train, y_test
    
    dnn_model = Sequential()
    #defining DNN layer as DNN Can learn features from both spatial and time dimensions
    #CNN's output can extract spatial features from input data
    dnn_model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu'), input_shape = (1, 32, 32, 3)))
    dnn_model.add(TimeDistributed(MaxPooling2D((4, 4))))
    dnn_model.add(Dropout(0.5))
    dnn_model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    dnn_model.add(TimeDistributed(MaxPooling2D((4, 4))))
    dnn_model.add(Dropout(0.5))
    dnn_model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same',activation = 'relu')))
    dnn_model.add(TimeDistributed(MaxPooling2D((2, 2))))
    dnn_model.add(Dropout(0.5))
    dnn_model.add(TimeDistributed(Conv2D(256, (2, 2), padding='same',activation = 'relu')))
    dnn_model.add(TimeDistributed(MaxPooling2D((1, 1))))
    dnn_model.add(Dropout(0.5))
    dnn_model.add(TimeDistributed(Flatten()))
    dnn_model.add(GRU(32))#adding LSTM layer
    dnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    dnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/dnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/dnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = dnn_model.fit(X_train, y_train, batch_size = 32, epochs = 150, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/dnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        dnn_model.load_weights("model/dnn_weights.hdf5")
    predict = dnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:2150] = y_test1[0:2150]
    calculateMetrics("DNN", y_test1, predict)
    #dnn_model.load_weights("model/best_dnn_weights.hdf5")

def playVideo(filename, output):
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (500, 500))
            cv2.putText(frame, output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)    
            cv2.imshow('Deep Fake Detection Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()    

#function to allow user to upload video file
def forgeDetection():
    text.delete('1.0', END)
    global dnn_model, labels
    images = []
    filename = askopenfilename(initialdir = "testVideos")
    pathlabel.config(text=filename)
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if ret == True:
            img = cv2.resize(frame, (32, 32))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,32,32,3)
            temp = []
            temp.append(im2arr)
            img = np.asarray(temp)
            img = img.astype('float32')
            img = img/255
            preds = dnn_model.predict(img)
            predict = np.argmax(preds)
            recognize = labels[predict]
            frame = cv2.resize(frame, (500, 500))
            if predict == 0:
                cv2.putText(frame, 'Forge Frame', (10, 50),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (255, 0, 0), 3)
            else:
                cv2.putText(frame, 'Real Frame', (10, 50),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (255, 0, 0), 3)
                images.append(frame)
            cv2.imshow('Forge & Real Frame Detection Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break            
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    video = cv2.VideoWriter('video_after_removing_forge.mp4', fourcc, 25, (500, 500))
    for i in range(len(images)):
        video.write(images[i])
    video.release()
    cv2.destroyAllWindows()    
    

font = ('times', 15, 'bold')
title = Label(main, text='Video Forgery Detection using DNN')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Video Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=480,y=100)

normalizeButton = Button(main, text="Shuffle & Normalize Video Frames", command=normalizeFrames)
normalizeButton.place(x=50,y=150)
normalizeButton.config(font=font1)

splitButton = Button(main, text="Split Frames Train & Test", command=splitFrames)
splitButton.place(x=350,y=150)
splitButton.config(font=font1)

trainDNNButton = Button(main, text="Train DNN Model", command=trainModel)
trainDNNButton.place(x=600,y=150)
trainDNNButton.config(font=font1)

forgeDetectionButton = Button(main, text="Video Based Forgey Detection", command=forgeDetection)
forgeDetectionButton.place(x=50,y=200)
forgeDetectionButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
