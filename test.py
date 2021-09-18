from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import cv2
import imutils as imu

json_file = open('model.json1', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model1.h5")
print("Loaded model from disk")
cm = cv2.VideoCapture("C:\\Users\\Bvaesh_Ram\\Downloads\\11hen.mp4")
firstframe =None 
area = 500

def classify(img_file):
    
    img_file = cv2.resize(img_file,(64,64))

    test_image = image.img_to_array(img_file)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    if result[0][0] == 1:
        prediction = 'Healthy'
        
        
    else:
        prediction = 'Neck Twisted'
        cv2.rectangle(img,(188,100),(500,400),(0,0,255),2)
        
    print(prediction)
    
    
    cv2.putText(img, prediction, (200, 100),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("",img)
    
    
    
  

while True:
    text = "No movement"
    _,img=cm.read()
    img=imu.resize(img, width=700)
    classify(img)
    key= cv2.waitKey(1) & 0XFF
    if key == ord('q'):
        break
cm.release()
cv2.destroyAllWindows()