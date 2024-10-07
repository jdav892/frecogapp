import cv2
from imgbeddings import imgbeddings
import numpy as np
import psycopg2
from PIL import Image

#loading the haar cascade algo from xml into opencv
alg = "/content/haarcascade_frontalface_default.xml"
#passing algo to opencv
haar_cascade = cv2.CascadeClassifier(alg)
#read the image as grayscale
file_name = '/content/lakers.jpg' #place face image here
#find the faces in that image
img = cv2.imread(file_name, 0)
#creating a black and white version of image
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#this gives back an array of face locations and sizes
faces = haar_cascade.detectMultiScale(
    gray_img,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(100,100)
)

i = 0
#for each face detected
for x, y , w, h in faces:
    #crop the image to select only the face
    cropped_image = img[y : y + h, x : x + w]
    #write the cropped image to a file (using stored-faces for file name here)
    target_file_name = 'stored-faces/' + str(i) + '.jpg' #insert the target image name here
    cv2.imwrite(
        target_file_name,
        cropped_image,
    )
    i = i + 1


#load image to search with
file_name2 = "/content/compare-img.jpg"
img2 = cv2.imread(file_name2, cv2.IMREAD_GRAYSCALE)
#find faces
faces2 = haar_cascade.detectMultiScale(
    gray_img,
    scaleFactor=1.05,
    minNeighbors=2,
    minSize=(100, 100),
)
#load 'imgbeddings' to calculate embeddings
ibed = imgbeddings()

#for each face in picture
for x, y, w, h in faces:
    cropped = img[y: y + h, x: x + w]
    if isinstance(cropped, np.ndarray):
        #crop the image to select for face
        cropped =Image.fromarray(cropped)
    #calculate the embeddings
    bron_img = ibed.to_embeddings(cropped)[0]