from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

model = load_model('detectfacemodel.h5')
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source = cv2.VideoCapture(0)

labels_dict = { 
    0: 'man',
    1 : 'woman',
}


while(True):

    ret,img = source.read()
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(img,1.3,5)  

    for x,y,w,h in faces:
        resized = img[y:y+h,x:x+w]
        resized = cv2.resize(resized,(128,128))
        resized = image.img_to_array(resized)
        resized = np.expand_dims(resized,axis=0)
        normalized = resized / 255.0
        result = model.predict_classes(normalized) 
        print('predicted value', result[0])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(
          img, labels_dict[result[0][0]], 
          (x, y-10),
          cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)        
    
    cv2.imshow('LIVE',img)
    key = cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()
