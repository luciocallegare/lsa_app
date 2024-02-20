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
import mediapipe as mp
import numpy as np
import json
from keras.models import load_model

MEASURES_MODEL = 224
N_FRAMES = 20
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
        static_image_mode= False,
        max_num_hands= 2,
        min_detection_confidence= 0.8,
        min_tracking_confidence = 0.5
    ) as hands:
    class CamApp(App):

        def build(self):
            self.pred_state = False
            # Main layout components 
            self.web_cam = Image(size_hint=(1,1), fit_mode = 'fill')
            self.button = Button(text="Empezar traducción",on_press = self.togglePrediction, size_hint=(1,.1))
            self.sign_label = Label(text="", size_hint=(1,.1))

            # Add items to layout
            layout = BoxLayout(orientation='vertical')
            layout.add_widget(self.web_cam)
            layout.add_widget(self.sign_label)
            layout.add_widget(self.button)

            # Load tensorflow/keras model
            self.model = load_model('./_internal/modelConvLSTM')
            self.threshold = 0.5
            f = open('./_internal/dataset.json',encoding='utf-8')
            self.label = json.load(f)
            # Setup video capture device
            self.capture = cv2.VideoCapture(0)
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = 33.0
            self.i = 0
            self.sequence = []
            self.lastSign = {}
            Clock.schedule_interval(self.update,1.0/self.fps)
            
            return layout

        def createHandsCanva(self,results):
            black_background = np.zeros((self.height,self.width,3),dtype = np.float32)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(black_background,
                                            hand_landmarks,
                                            mp_hands.HAND_CONNECTIONS,
                                            #mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                            #mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=10, circle_radius=2)
                                            )
            return black_background
        
        def togglePrediction(self, *args):
            self.pred_state = not self.pred_state
            if self.pred_state:
                self.button.text = "Finalizar traducción"
            else:
                self.button.text = "Empezar traducción"
                self.sign_label.text = ""
                self.i = 0
                self.lastSign = {}
                self.sequence = []

        # Run continuously to get webcam feed
        def update(self, *args):

            # Read frame from opencv
            ret, frame = self.capture.read()
            frame = cv2.flip(frame,1)
            frame =cv2.flip(frame, 0)
            # Change to rgb
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Flip horizontall and convert image to texture
            buf = frame.tobytes()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.web_cam.texture = img_texture
            if self.pred_state:
                if self.lastSign:
                    self.sign_label.text = f"{self.lastSign['name']}:{self.lastSign['percentage']}%"
                video_frames_count = self.fps*2
                skip_frames_window = int(video_frames_count/N_FRAMES)
                results = hands.process(frame)
                if self.i < skip_frames_window:
                    self.i+=1
                    return
                if results.multi_hand_landmarks is not None:
                    handsCanva = self.createHandsCanva(results)
                    handsCanva = cv2.cvtColor(handsCanva, cv2.COLOR_BGR2GRAY)
                    handsCanva =  cv2.resize(handsCanva,(MEASURES_MODEL,MEASURES_MODEL))
                    self.sequence.append(handsCanva)
                else:
                    self.i = 0
                    self.sequence = []
                    self.lastSign = {}
                if len(self.sequence) == N_FRAMES:
                    cantHands = len(results.multi_hand_landmarks)
                    probArray =  self.model.predict(np.expand_dims(self.sequence,0))
                    idGesture =  np.argmax(probArray)
                    probPred = probArray[0][idGesture]
                    cantHandsPred = 1 if self.label[idGesture]['handsUsed'] != 'B' else 2
                    print(idGesture,probPred)
                    if probPred > self.threshold and cantHands == cantHandsPred:
                        self.lastSign = {
                            "name":self.label[idGesture]["name"],
                            "percentage":round(probPred*100,2)
                        }
                        print(self.label[idGesture]["name"])
                    else:
                        self.lastSign = {}
                    self.sequence = []                        


    if __name__ == '__main__':
        CamApp().run()