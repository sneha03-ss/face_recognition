import cv2
import face_recognition
import pickle
import os

folderPath = 'faces'
pathList = os.listdir(folderPath)
#print(pathList)

imgList = []
faceNames = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    faceNames.append(os.path.splitext(path)[0])

    #print(os.path.splitext(path)[0])

print(faceNames)

def findEncoding(imageList):
    
    encodeList = []
    for img in imageList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

print("Encoding Started...")
encodeListKnown = findEncoding(imgList)
encodeListKnownWithNames = [encodeListKnown, faceNames]
print("Encoding Complete")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithNames, file)
file.close()
print("File Saved")
