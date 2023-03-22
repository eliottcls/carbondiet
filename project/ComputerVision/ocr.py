from paddleocr import PaddleOCR
from project.ComputerVision.box_detection import BoundingBoxDetection

class OCR:

    def __init__(self):
        self.paddleocr = PaddleOCR(show_log=False, use_angle_cls=True)
        self.boxdetect = BoundingBoxDetection()

    def extract_text_from_one_box(self, image, box):
        x, y, w, h = box
        result = self.paddleocr.ocr(image[y:y+h,x:x+w], cls=True)
        list_formated_result = [x[1][0] for x in result[0]]
        print(list_formated_result)
        return (' '.join(list_formated_result))

    def extract_text_from_dict_of_box(self, image, box_dict):
        classes = list(box_dict.keys())
        dict_results = {x:[] for x in classes}

        for cls in classes:
            for box in box_dict[cls]:
                dict_results[cls].append(self.extract_text_from_one_box(image, box))
        
        return dict_results
    
    def extract_text_from_menu(self, image):

        boxes = self.boxdetect.detect_boxes(image)

        result = self.extract_text_from_dict_of_box(image, boxes)

        return result