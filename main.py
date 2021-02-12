# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

files = glob.glob("output/*.png")
for f in files:
    os.remove(f)

from sort import *

tracker = Sort()

line = [(43, 543), (550, 655)]
counter = 0
ids = []

points = [[43, 543], [480, 330], [620, 350], [580, 550]]
counter = 0
pts = pts = np.array([[43, 543], [480, 330], [620, 350], [580, 550]], np.int32)
pts = pts.reshape((-1, 1, 2))
isClosed = True


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# load the COCO class labels our YOLO model was trained on
labelsPath = "C:\\Users\\NIU2KOR\\Desktop\\learning\\vehicle_occupancy_and_tracking\\yolo-coco\\coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
print(LABELS)
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture("input/highway.mp4")
writer = None
(W, H) = (None, None)

frameIndex = 0


# try to determine the total number of frames in the video file
try:
    prop = (
        cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    )
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x + w, y + h, confidences[i]])

    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3], track[4]])
        indexIDs.append(int(track[4]))

    if len(boxes) > 0:
        i = int(0)

        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
            p1 = (int((x + (w - x) / 2) + 30), int((y + (h - y) / 2) + 30))
            cv2.line(frame, p0, p1, color, 1)

            # check if center points of object is inside the polygon
            point = Point((int(x + (w - x) / 2), int(y + (h - y) / 2)))
            polygon = Polygon(points)
            if (polygon.contains(point)) == True and box[4] not in ids:
                counter += 1
                ids.append(box[4])
            # decrease counter while passing out of the polygon
            if intersect(p0, p1, points[0], points[3]) and box[4] in ids:
                counter = counter - 1
                ids.remove(box[4])
            text = "{}".format(indexIDs[i])
            cv2.putText(
                frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            i += 1

    # draw line
    cv2.polylines(frame, [pts], isClosed, (0, 255, 255), 1)
    # draw counter
    cv2.putText(
        frame, str(counter), (100, 200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10
    )
    # saves image file
    cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            "out.mp4", fourcc, 30, (frame.shape[1], frame.shape[0]), True
        )

        # some information on processing single frame
        if total > 0:
            elap = end - start
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    # write the output frame to disk
    writer.write(frame)
    # increase frame index
    frameIndex += 1

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
