# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Import haarcascades
frontalFaceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
profileFaceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_profileface.xml")

# Build app and layout 
class CamApp(App):

    def build(self):
        # Main layout components 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def detectFace(self, grayscale, image):
        imageDet = image
        
        # Detects frontal faces in the image using the face cascade
        faces = frontalFaceCascade.detectMultiScale(
            grayscale,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30,30),
        )

        # Detects profile faces in the image using the face cascade
        profileFaces = profileFaceCascade.detectMultiScale(
            grayscale,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30,30),
        )

        # Detect profile faces in the flipped image to detect profile faces facing right
        flipped = cv2.flip(grayscale, 1)
        profileFacesFlipped = profileFaceCascade.detectMultiScale(
            flipped,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30,30),
        )

        if len(faces) != 0:
            for (x, y, w, h) in faces:
                imageDet = image[y:y+h, x:x+w]
        elif len(profileFaces) != 0:
            for (x, y, w, h) in profileFaces:
                imageDet = image[y:y+h, x:x+w]
        elif len(profileFacesFlipped) != 0:
            for (x, y, w, h) in profileFacesFlipped:
                imageDet = image[y:y+h, x:x+w]

        imageDet = cv2.resize(imageDet, (250, 250))

        return imageDet

    # Load image from file and conver to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        # Scale image to be between 0 and 1 
        img = img / 255.0
        
        # Return image
        return img

    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.65
        verification_threshold = 0.75

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = self.detectFace(grayscale, frame)

        cv2.imwrite(SAVE_PATH, image)

        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold

        # Set verification text 
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        
        return results, verified



if __name__ == '__main__':
    CamApp().run()