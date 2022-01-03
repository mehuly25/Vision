import cv2


#for a single image
image=cv2.imread("img.jpg")

classNames=[]
classFile='coco.names'
with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
#print(classNames)

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds,confs,bbox=net.detect(image,confThreshold=0.5)
print(classIds)

for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
    print(classId, classNames[classId-1])
    cv2.rectangle(image,box,color=(255,255,0),thickness=3)
    cv2.putText(image,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)

cv2.imshow("Output", image)
cv2.waitKey(0)
