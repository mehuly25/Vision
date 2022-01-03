import pyttsx3
import cv2
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def talk(text):
    engine.say(text)
    engine.runAndWait()


cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
#cap.set(10,70)

thres=0.45
classNames = []
classFile = "coco.names"
with open(classFile,'rt') as f:
   classNames = [line.rstrip() for line in f]

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
   success,img = cap.read()
   classIds, confs, bbox = net.detect(img, confThreshold=thres)
   #print(classIds, bbox)

   if len(classIds) != 0:
       for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
           print(classId,classNames[classId-1])
           cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
           cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
           cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
           talk(classNames[classId-1])
   cv2.imshow("Output",img)
   cv2.waitKey(1)

'''
camera.release()
cv2.destroyAllWindows()
'''