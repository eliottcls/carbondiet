from flask import Flask, request, jsonify
from project.ComputerVision.ocr import OCR
import cv2
import numpy as np 

app = Flask(__name__)
ocr = OCR()

def create_opencv_image_from_stringio(img_stream):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

@app.route('/menutext', methods=['POST'])
def upload():
    img = create_opencv_image_from_stringio(request.files['image'].stream)
    result = ocr.extract_text_from_menu(img)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False)