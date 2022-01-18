import cv2
import os
import numpy as np


dataPath='C:/Users/jorge/Documents/mascarilla/Dataset_faces'
dir_list=os.listdir(dataPath)
print("listade archivos", dir_list)

labels=[]
faceData=[]
label=0

for name_dir in dir_list:
    dir_path=dataPath + "/" + name_dir
    
    for file_name in os.listdir(dir_path):
        image_path=dir_path + "/" + file_name
        print(image_path)
        image=cv2.imread(image_path,0)
        #cv2.imshow("image",image)
        #cv2.waitKey(10)
        faceData.append(image)
        labels.append(label)
    label +=1
    

facemask=cv2.face.LBPHFaceRecognizer_create()

print("entrenando modelo")
facemask.train(faceData,np.array(labels))

facemask.write("face_mask_model.xlm")
print("modelo guardado")