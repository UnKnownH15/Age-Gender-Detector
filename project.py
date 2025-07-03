import cv2
import argparse

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)','(13-17)', '(18-19)','(20-21)','(22-24)','(25-38)','(29-30)', '(33-43)', '(48-60)', '(60-100)']
genderList = ['Male', 'Female']

def load_models():
    faceNet = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
    ageNet = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
    genderNet = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
    return faceNet, ageNet, genderNet

def highlightFace(net, frame, conf_threshold=0.7):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, 8)
    return frame, faceBoxes

def predict_age_gender(face, ageNet, genderNet):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    gender = genderList[genderNet.forward()[0].argmax()]
    ageNet.setInput(blob)
    age = ageList[ageNet.forward()[0].argmax()]
    return gender, age

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help="Path to image or video file")
    parser.add_argument('--camera', type=int, help="Camera index (0 = default)", default=None)
    args = parser.parse_args()

    faceNet, ageNet, genderNet = load_models()


    if args.image:
        video = cv2.VideoCapture(args.image)
    elif args.camera is not None:
        video = cv2.VideoCapture(args.camera)
    else:
        video = cv2.VideoCapture(2)  #You can Try diff numbers if it doesn't show your desired input webcam

    padding = 20
    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        resultImg, faceBoxes = highlightFace(faceNet, frame)

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                         max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

            gender, age = predict_age_gender(face, ageNet, genderNet)
            label = f"{gender}, {age}"
            print(f"Gender: {gender}, Age: {age}")
            cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Detecting age and gender", resultImg)

if __name__ == "__main__":
    main()
