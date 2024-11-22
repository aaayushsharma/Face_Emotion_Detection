import cv2
import os
import numpy as np
from keras.models import model_from_json, Sequential

# Disable OneDNN optimizations if necessary
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the model from JSON
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()

# Use custom_objects to properly handle `Sequential`
model = model_from_json(model_json, custom_objects={'Sequential': Sequential})

# Load the weights into the model
model.load_weights("facialemotionmodel.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract and preprocess image features
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start webcam capture
webcam = cv2.VideoCapture(0)

# Emotion labels corresponding to model predictions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

try:
    while True:
        # Capture frame from the webcam
        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Loop through detected faces and make predictions
        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]  # Extract face region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around the face
            
            # Resize the face to 48x48 pixels (input size for the model)
            face_img = cv2.resize(face_img, (48, 48))
            
            # Preprocess the face image for model prediction
            img = extract_features(face_img)
            
            # Make prediction
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            # Display the predicted emotion label
            cv2.putText(frame, prediction_label, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        
        # Display the frame with detected faces and emotion predictions
        cv2.imshow("Emotion Detection", frame)
        
        # Press 'Esc' to exit the loop
        if cv2.waitKey(27) & 0xFF == 27:
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release the webcam and close OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()
