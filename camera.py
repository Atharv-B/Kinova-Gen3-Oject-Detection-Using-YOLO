#Imports
import numpy as np
import torch
import cv2
import time

from videocaptureasync import VideoCaptureAsync

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient

from kortex_api.SessionManager import SessionManager


from kortex_api.autogen.messages import Session_pb2

class ObjectDetetction:
    def __init__(self, stream):
        self.stream = stream
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    
    def main(self):
        DEVICE_IP = "192.168.1.10"
        DEVICE_PORT = 10000

        # Setup API
        errorCallback = lambda kException: print("_________ callback error _________ {}".format(kException))

        #transport = UDPTransport()
        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, errorCallback)
        self.transport.connect(DEVICE_IP, DEVICE_PORT)

        # Create session
        print("Creating session for communication")
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = 'admin'
        session_info.password = 'admin'
        session_info.session_inactivity_timeout = 60000   # (milliseconds)
        session_info.connection_inactivity_timeout = 2000 # (milliseconds)
        print("Session created!")
        self.session_manager = SessionManager(self.router)   
        self.session_manager.CreateSession(session_info)
    
    def get_video(self):
        return VideoCaptureAsync(self.stream)

    def load_model(self):
        '''
            Load the YOLO (pretranied) model.
        '''
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def get_frame(self, frame):
        '''
            Takes a single frame input and scores the frame using the YOLO5 model. 
            Returns Labels and Coordinates of objects detected by the model in the frame.
        '''
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels = results.xyxyn[0][:,-1].cpu().numpy()
        cords = results.xyxyn[0][:,:-1].cpu().numpy()
        return labels, cords
    
    def class_to_label(self, x):
        '''
            For a given label value, returns the corresponding string label.
        '''
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame):
        '''
            This Function will plot boxes over the objects in the Frame which has score above the mentioned threshold (0.2).
        '''
        labels, cords = results
        x_shape = frame.shape[1]
        y_shape = frame.shape[0]
        for i in range(len(labels)):
            row = cords[i]
            if row[4] >= 0.2:
                x1 = int(row[0]*x_shape)
                x2 = int(row[2]*x_shape)
                y1 = int(row[1]*y_shape)
                y2 = int(row[3]*y_shape)
                bgr = (0,255,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame
    
    def __call__(self):
        '''
            This funciton is called when class is executed. It runs the loop to read the input frame by frame.
        '''
        self.main()
        player = self.get_video()
        player.start()
        while True:
            ret, frame = player.read()
            start_time = time.time()
            assert ret
            results = self.get_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time.time()
            fps = 1/np.round(end_time - start_time, 3)
            print("FPS-{}".format(fps))
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        player.stop()
        cv2.destroyAllWindows()

        print('Closing Session..')
        self.session_manager.CloseSession()
        self.router.SetActivationStatus(False)
        self.transport.disconnect()
        print('Done!')

#Create and object and call.
test = ObjectDetetction("rtsp://192.168.1.10/color")
test()
