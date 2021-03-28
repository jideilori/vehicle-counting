# This  makes use of video - https://www.youtube.com/watch?v=MNn9qKG2UFI&t=3s

import cv2
import numpy as  np
import dlib
trackers=[]

# Initialize the parameters
confThreshold = 0.5 #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

# Load names of classes
classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov4-tiny.cfg";
modelWeights = "yolov4-tiny.weights";
# modelConfiguration = "yolov4.cfg";
# modelWeights = "yolov4.weights";


net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
##net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
def findCenter(x,y,w,h):
    cx = int((x+w)/2)
    cy = int((y+h)/2)
    cv2.circle(frame_cropped, (cx, cy),2, (25,250,250), -1)

    return cx,cy


def pointInRect(x,y,w,h,cx,cy):
    x1, y1 = cx,cy
    if (x < x1 and x1 < x+w):
        if (y < y1 and y1 < y+h):
            return True
    else:
        return False
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

cap = cv2.VideoCapture('../highway2.mp4')

    # Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]



# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    global inCount,Font,count,SKIP_FRAMES,outCount

    frameHeight = frame_cropped.shape[0]
    frameWidth = frame_cropped.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    trackers_to_del=[]
    # Delete lost trackers based on tracking quality
    for tid,trackersid in enumerate(trackers):
        trackingQuality = trackersid[0].update(frame_cropped)
        if trackingQuality < 5:
            trackers_to_del.append(trackersid[0])
    try:
        for trackersid in trackers_to_del:
            trackers.pop(tid)
    except IndexError:
        pass

    

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        classId, conf, left, top, right, bottom = classIds[i], confidences[i], left, top, left + width, top + height
        
        rect = dlib.rectangle(left,top,right,bottom)
        (x,y,w,h)= rect_to_bb(rect)

        tracking = False

        for trackersid in trackers:
            pos = trackersid[0].get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            tx,ty=findCenter(startX,startY,endX,endY)

            t_location_chk = pointInRect(x,y,w,h,tx,ty)
            if t_location_chk:
                tracking = True
            
        if not tracking:
            tracker = dlib.correlation_tracker()
            tracker.start_track(frame_cropped, rect)
            trackers.append([tracker,frame_cropped])

    for num,trackersid in enumerate(trackers):                        
        pos = trackersid[0].get_position()
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())
        
        cv2.rectangle(frame_cropped, (startX, startY), (endX, endY),(0, 255, 250), 1)       
        if endX< 380 and endY>=280:
            inCount+=1
            trackers.pop(num)

        
        

inCount= 0 
outCount = 0
Font=cv2.FONT_HERSHEY_COMPLEX_SMALL
while True :
   
    # get frame from the video
    ret, frame = cap.read()
    frame_o = cv2.resize(frame,(640,480))
    # frame_cropped = frame_o[200:640,0:640]
    frame_cropped=frame_o[200:640,0:380]


    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame_cropped, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame_cropped, outs)
    
    cv2.putText(frame_o,f"IN:{inCount}",
                    (20,40),Font,1,(255,0,0),2)


    cv2.imshow('frame_o',frame_o)
    cv2.imshow('frame_cropped',frame_cropped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
