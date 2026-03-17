
import cv2
import numpy as np
import face_recognition
import pickle

cap = cv2.VideoCapture(0)

print("Loading Encode File...") 
file = open('EncodeFile.p', 'rb')
encodeListKnownWithNames = pickle.load(file)
file.close()
encodeListKnown, faceNames = encodeListKnownWithNames
print("Encode File Loaded...")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame")
        continue

    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations(imgs)
    encodeCurrFrame = face_recognition.face_encodings(imgs, faceCurrFrame) 

    for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDistance)

        top, right, bottom, left = faceLoc
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4  

        if matches[matchIndex] and faceDistance[matchIndex] < 0.6:  
            name = faceNames[matchIndex]
            color = (0, 255, 0)  
            label = f"{name} ({(1 - faceDistance[matchIndex]) * 100:.2f}%)"
        else:
            name = "Unknown"
            color = (0, 0, 255)  
            label = "Unknown"

        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
