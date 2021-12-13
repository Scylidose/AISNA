import face_recognition
from imutils import paths
import pickle
import cv2
from random import randrange

imgLink = "data/img/"
imagePaths = list(paths.list_images(imgLink))

# Get user supplied values
cascPath = "config/haarcascade_frontalface_default.xml"

def encodeFaces():
    # get paths of each file in folder named Images
    # Images here contains my data(folders of various persons)
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

cascPathface = "config/haarcascade_frontalface_alt2.xml"

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# load the harcaascade in the cascade classifier
# faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('config/face_enc', "rb").read())

def liveRecognition():
    # start streaming video from webcam
    video_capture = cv2.VideoCapture(0)

    # label for video
    label_html = 'Capturing...'
    # initialze bounding box to empty
    bbox = ''
    count = 0 
    count_images = len(imagePaths)

    irrelevant = "UNKNOWN"

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(60, 60),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    
        # convert the input frame from BGR to RGB 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # the facial embeddings for face in input
        encodings = face_recognition.face_encodings(rgb)
        names = []
        # loop over the facial embeddings incase
        # we have multiple embeddings for multiple fcaes
        for encoding in encodings:
        #Compare encodings with encodings in data["encodings"]
        #Matches contain array with boolean values and True for the embeddings it matches closely
        #and False for rest
            matches = face_recognition.compare_faces(
                data["encodings"],
                encoding)
            #set name =inknown if no encoding matches
            name = irrelevant
            # check to see if we have found a match
            if True in matches:
                #Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    #Check the names at respective indexes we stored in matchedIdxs
                    name = "ADMIN"
                    #increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                #set name which has highest count
                name = max(counts, key=counts.get)
    
            f = open("transcription.txt", "r")

            transcript = f.read().splitlines()
            transcript1 = ""
            transcript2 = ""

            if len(transcript) >= 1:
                transcript1 = transcript[0]

                if len(transcript) >= 2:
                    transcript2 = transcript[1]

            # update the list of names
            names.append(name)

            print(names)
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):
                random_take = randrange(50)

                # rescale the face coordinates
                # draw the predicted face name on the image
                if name == "ADMIN" and random_take == 5:
                    cv2.imwrite(imgLink+'admin-'+str(count_images+1)+'.jpg',frame)
                    print("Saving...")
                    count_images += 1

                if transcript1 == "hello":
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 250, 251), 2)

                    if transcript2 == "come on":
                        cv2.putText(frame, name, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, (0,250,251))

                        if name == irrelevant:
                            cv2.putText(frame, name, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, (255, 255, 255))

                    if name == irrelevant:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()