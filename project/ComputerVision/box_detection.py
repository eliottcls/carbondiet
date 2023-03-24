from ultralytics import YOLO
import os
from dotenv import load_dotenv

load_dotenv()

class BoundingBoxDetection:

    def __init__(self):
        self.model = YOLO(os.environ.get('BEST_MODEL'))
        self.results = None


    def detect_boxes(self, image, prob=True):

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

    #TODO create function to delete overlapped boxes -> to not have twice the same recipe
    #TODO integrate this function in detect boxes

    def remove_overlapping_bounding_boxes(self, bounding_boxes, treshold=0.5):
        #input a list of tuple containing bounding boxes and confidence
        cleaned_boxes = {}
        for box, prob in bounding_boxes:
            x1, y1, w1, h1 = box
            area1 = w1 * h1
            added = False
            for key in cleaned_boxes:
                x2, y2, w2, h2 = cleaned_boxes[key][0]
                area2 = w2 * h2
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap
                if overlap_area > treshold * area1 or overlap_area > treshold * area2:
                    if prob > cleaned_boxes[key][1]:
                        cleaned_boxes[key] = (box, prob)
                    added = True
                    break
            if not added:
                cleaned_boxes[len(cleaned_boxes)] = (box, prob)
        return list(cleaned_boxes.values())
    
    #TODO create function to locate recipe regarding section title