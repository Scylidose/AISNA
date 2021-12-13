import face_recognition
from imutils import paths
import pickle
import cv2

imgLink = "data/img/"

# Get user supplied values
imagePath = "child.png"
cascPath = "config/haarcascade_frontalface_default.xml"

def encodeFaces():
    # get paths of each file in folder named Images
    # Images here contains my data(folders of various persons)
    imagePaths = list(paths.list_images(imgLink))
    knownEncodings = []
    knownNames = []
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = "ADMIN"
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Use Face_recognition to locate faces
        boxes = face_recognition.face_locations(rgb, model='hog')
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    # save emcodings along with their names in dictionary data
    data = {"encodings": knownEncodings, "names": knownNames}
    # use pickle to save data into a file for later use
    f = open("config/face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()


def faceIdentification():
    data = pickle.loads(open("config/face_enc", "rb").read())

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    irrelevant = "IRRELEVANT"
    # Read the image
    image = cv2.imread(imagePath)
    # convert the input frame from BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        rgb,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30))

    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)

    print("Found {0} faces!".format(len(faces)))

    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = irrelevant
        if True in matches:
            # Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                # Check the names at respective indexes we stored in matchedIdxs
                name = "ADMIN"
                # increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
            # set name which has highest count
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

        # Draw a rectangle around the faces
        for ((x, y, w, h), name) in zip(faces, names):

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 250, 251), 2)

            cv2.putText(
                image, name, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 251), 2)

            if name == irrelevant:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)

                cv2.putText(
                image, name, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    image = cv2.resize(image, (540, 960))
    cv2.imshow("Frame", image)
    cv2.waitKey(0)

faceIdentification()