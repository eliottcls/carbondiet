from ultralytics import YOLO
import os
from dotenv import load_dotenv

load_dotenv()

class BoundingBoxDetection:

    def __init__(self):
        self.model = YOLO(os.environ.get('BEST_MODEL'))
        self.results = None


    def detect_boxes(self, image):

        #run model
        if self.results:
            pass
        else:
            self.results = self.model(source=image)

        #extract infos
        list_classes = [self.results[0].names[int(x.item())] for x in self.results[0].boxes.cls]
        list_boxes = self.results[0].boxes.xywh.tolist()
        
        dict_results = {x:[] for x in self.results[0].names.values()}

        for i in range(len(list_classes)):
            dict_results[list_classes[i]].append(self.normalize_coordinates(list_boxes[i]))

        return dict_results

    def normalize_coordinates(self, coord):
        # Function to pass from yolov8 coordinate format to opencv format
        x1, y1, w, h = coord
        x = x1 - (w / 2)
        y = y1 - (h / 2)

        return(int(x), int(y), int(w), int(h))
    
    def display_boxes(self, image):
        
        if self.results:
            pass
        else:
            self.results = self.model(source=image)

        return self.results[0].plot()
