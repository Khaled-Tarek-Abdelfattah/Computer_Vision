import numpy as np
import time
import cv2
import os
import glob
import matplotlib.pyplot as plt
import sys

weights_path = os.path.join("yolo","yolov3.weights")
config_path = os.path.join("yolo","yolov3.cfg")

net = cv2.dnn.readNetFromDarknet(config_path,weights_path)

names = net.getLayerNames()

layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()]

def process_image(img):
    (H, W) = img.shape[:2]
    layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), crop=False, swapRB=False)
    net.setInput(blob)
    layers_output = net.forward(layers_names)

    boxes = []
    confidences = []
    classIDs =[]

    for output in layers_output:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if(confidence > 0.85):
                box = detection[:4] * np.array([W, H, W, H])
                bx, by, bw, bh = box.astype("int")

                x = int(bx - (bw / 2))
                y = int(by - (bh / 2))

                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.8)

    labels_path = os.path.join("yolo", "coco.names")
    labels = open(labels_path).read().strip().split("\n")

    for i in idx.flatten():
        (x, y) = [boxes[i][0], boxes[i][1]]
        (w, h) = [boxes[i][2], boxes[i][3]]

        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, "{}: {}".format(labels[classIDs[i]], confidences[i]), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,\
                   0.5, (255,0,0), 2)
    return img

print(sys.argv)

cap = cv2.VideoCapture(sys.argv[1])
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_cod = cv2.VideoWriter_fourcc(*'DIVX')
video_output= cv2.VideoWriter(sys.argv[2],
                      video_cod, 20,
                      (width,height))
while(cap.isOpened()):
    _, frame = cap.read()

    processed_frame = process_image(frame)

    video_output.write(processed_frame)

    cv2.imshow('frame',processed_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
