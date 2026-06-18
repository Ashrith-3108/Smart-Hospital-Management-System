import cv2
import numpy as np
import pytesseract

'''

import os
import cv2

labels = ['Aadhar_no', 'DOB', 'Gender', 'Name']
#yolo confidence threshold to detect hand signs
CONFIDENCE_THRESHOLD = 0.50
GREEN = (0, 255, 0)
yolo_model = YOLO("/content/runs/detect/train/weights/best.pt")
print("Yolo11 Model Loaded")
#function to detect signs using YOLO11
def detectSign(frame):
    global yolo_model
    detections = yolo_model(frame)[0]
    label = None
    # loop over the detections
    for data in detections.boxes.data.tolist():
        print(data)
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]
        cls_id = data[5]
        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) >= CONFIDENCE_THRESHOLD:
          xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
          label = labels[int(cls_id)]
          print(str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)+" "+label)
          cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)
          cv2.putText(frame, labels[int(cls_id)], (xmin, ymin-10),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)            
    return frame, label

import matplotlib.pyplot as plt
img = cv2.imread("/content/download1.jpg")
img = cv2.resize(img, (500, 500))
frame, label = detectSign(img)
plt.imshow(frame)
plt.show()    
'''
img = cv2.imread("samples/download1.jpg")
img = cv2.resize(img, (500, 500))

boxes = []
#boxes.append([134, 397, 225, 424]) amee
#boxes.append([94, 255, 260, 290])

#boxes.append([159,301,356,374])download

#download1
boxes.append([174, 414, 272, 426])
boxes.append([162, 268, 340, 291])

for i in range(len(boxes)):
    box = boxes[i]
    region = img[box[1]:box[3], box[0]:box[2]]
    text = pytesseract.image_to_string(region)
    print(text)
    cv2.imshow("region", region)
    cv2.waitKey(0)
    max_mask = box[0] + int(box[2] / 3)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    for y in range(box[1], box[3]):
        for x in range(box[0], max_mask):
            img[y, x] = [0, 0, 0]    

cv2.imshow("aa", img)
cv2.waitKey(0)
