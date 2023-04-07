from paddleocr import PaddleOCR
from project.ComputerVision.box_detection import BoundingBoxDetection
from ultralytics import YOLO
import os
from dotenv import load_dotenv
import cv2
import math


load_dotenv()


class OCR:

    def __init__(self):
        self.paddleocr = PaddleOCR(show_log=False, use_angle_cls=True, enable_mkldnn=True, lang="latin")
        self.model = YOLO(os.environ.get('BEST_MODEL'))
        self.results = None
        self.cleaned_dict_of_boxes = None

    def extract_text_from_one_box(self, image, box):
        x, y, w, h = box
        result = self.paddleocr.ocr(image[y:y+h,x:x+w], cls=True)
        list_formated_result = [x[1][0] for x in result[0]]
        return (' '.join(list_formated_result))

    def extract_text_from_dict_of_box(self, image, box_dict):
        classes = list(box_dict.keys())
        dict_results = {x:[] for x in classes}

        for cls in classes:
            for box in box_dict[cls]:
                dict_results[cls].append(self.extract_text_from_one_box(image, box))
        
        return dict_results
    
    def extract_text_from_list_of_box(self, image, box_list):
        list_result = []
        for box in box_list:
            list_result.append(self.extract_text_from_one_box(image, box))
        
        return list_result
    
    def extract_text_from_menu(self, image):

        boxes = self.return_cleaned_boxes(image)
        list_section_title = boxes["Section title"]

        attributed_boxes = self.attribute_boxes(boxes)

        dict_text = self.extract_text_from_dict_of_box(image, attributed_boxes)
        section_title_text = self.extract_text_from_list_of_box(image, list_section_title)

        dict_result = {}

        for k in dict_text.keys():
            if k!="unassigned":
                dict_result[section_title_text[int(k)]] = dict_text[k]
            else:
                dict_result["Unassigned Recipe"] = dict_text[k]

        return dict_result
    
    def detect_boxes(self, image, prob=False):

        #run model
        self.results = self.model(source=image)

        #extract infos
        list_classes = [self.results[0].names[int(x.item())] for x in self.results[0].boxes.cls]
        list_boxes = self.results[0].boxes.xywh.tolist()
        list_confidence = self.results[0].boxes.conf.tolist()
        
        dict_results = {x:[] for x in self.results[0].names.values()}

        if prob:
            for i in range(len(list_classes)):
                dict_results[list_classes[i]].append((self.normalize_coordinates(list_boxes[i]), list_confidence[i]))
        else:
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
        #output a cleaned (according to fixed treshold) list of tuple containing bounding boxes and confidence
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
    
    def return_cleaned_boxes(self, image, treshold = 0.5):

        raw_boxes = self.detect_boxes(image, prob=True)
        cleaned_dict_of_boxes = {}

        for key in raw_boxes.keys():
            list_raw_boxes = [(x[0], x[1]) for x in raw_boxes[key]]
            list_cleaned_boxes = [x[0] for x in self.remove_overlapping_bounding_boxes(list_raw_boxes, treshold)]
            cleaned_dict_of_boxes[key] = list_cleaned_boxes
        
        return cleaned_dict_of_boxes

    def return_results(self, image):
        if self.results:
            pass
        else:
            self.results = self.model(source=image)

        return self.results
    
    def display_cleaned_boxes(self, image):
        if self.cleaned_dict_of_boxes:
            pass
        else:
            self.cleaned_dict_of_boxes = self.return_cleaned_boxes(image)
        
        list_all_boxes = []
        [list_all_boxes.extend(x) for x in self.cleaned_dict_of_boxes.values()]
        for box in list_all_boxes:
            x, y, w ,h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return image
    
    def attribute_boxes(self, dict_boxes, recipe_name = "Recipe", section_title_name = "Section title"):
        # Function that takes a dict of boxes with recipes and section itles. And then assigns each recipe to the closest section title box located above.

        recipe = dict_boxes[recipe_name]
        section = dict_boxes[section_title_name]
        
        # Initialize dictionary to hold assigned boxes
        assigned_boxes = {}

        # Initialize list to hold unassigned recipe boxes
        unassigned_boxes = []

        # Sort recipe boxes by y-coordinate
        recipe = sorted(recipe, key=lambda box: box[1])

        # Loop over recipe boxes
        for recipe_box in recipe:
            # Initialize variables to keep track of closest section box and distance
            closest_section_box = None
            closest_distance = float('inf')

            # Loop over section boxes
            for section_box in section:
                # Check if section box is above recipe box
                if section_box[1] > recipe_box[1]:
                    continue
                
                # Compute distance between centers of recipe box and section box
                recipe_center = (recipe_box[0] + recipe_box[2]/2, recipe_box[1] + recipe_box[3]/2)
                section_center = (section_box[0] + section_box[2]/2, section_box[1] + section_box[3]/2)
                distance = math.sqrt((recipe_center[0] - section_center[0])**2 + (recipe_center[1] - section_center[1])**2)
                # Check if distance is smaller than current closest distance
                if distance < closest_distance:
                    closest_section_box = section_box
                    closest_distance = distance

            # Check if a section box was found for the recipe box
            if closest_section_box is not None:
                # Add recipe box to dictionary of assigned boxes
                if section.index(closest_section_box) in assigned_boxes:
                    assigned_boxes[section.index(closest_section_box)].append(recipe_box)
                else:
                    assigned_boxes[section.index(closest_section_box)] = [recipe_box]
            else:
                # Add recipe box to list of unassigned boxes
                unassigned_boxes.append(recipe_box)

        # Add unassigned boxes to dictionary of assigned boxes
        assigned_boxes['unassigned'] = unassigned_boxes

        return assigned_boxes
    
    #TODO create function to locate recipe regarding section title