import streamlit as st
import PIL
import cv2
import numpy as np
import pandas as pd
import torch
import os
import io
# import sys
# import json
from collections import OrderedDict, defaultdict
import xml.etree.ElementTree as ET
from tempfile import TemporaryDirectory
import xlsxwriter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch

from paddleocr import PaddleOCR
# import pytesseract
# from pytesseract import Output
from fitz import Rect

import postprocess


st.set_page_config(page_title='Table Extraction Demo', layout='wide')


@st.experimental_singleton(ttl=3600)
def load_ocr_instance():
    ocr_instance = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True)
    return ocr_instance


@st.experimental_singleton(ttl=3600)
def load_detection_model():
    detection_model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/detection_wts.pt', force_reload=True, skip_validation=True, trust_repo=True)
    return detection_model


@st.experimental_singleton(ttl=3600)
def load_structure_model():
    structure_model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/structure_wts.pt', force_reload=True, skip_validation=True, trust_repo=True)
    return structure_model


ocr_instance, detection_model, structure_model = load_ocr_instance(), load_detection_model(), load_structure_model()

detection_class_names = ['table', 'table rotated', 'no object']
structure_class_names = [
    'table', 'table column', 'table row', 'table column header',
    'table projected row header', 'table spanning cell', 'no object'
]

detection_class_map = {k: v for v, k in enumerate(detection_class_names)}
structure_class_map = {k: v for v, k in enumerate(structure_class_names)}

detection_class_thresholds = {
    'table': 0.5,
    'table rotated': 0.5,
    'no object': 10
}
structure_class_thresholds = {
    "table": 0.45,
    "table column": 0.6,
    "table row": 0.5,
    "table column header": 0.4,
    "table projected row header": 0.3,
    "table spanning cell": 0.5,
    "no object": 10
}


def PIL_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv_to_PIL(cv_img):
    return PIL.Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def table_detection(pil_img, imgsz=640):
    image = PIL_to_cv(pil_img)
    pred = detection_model(image, size=imgsz)
    pred = pred.xywhn[0]
    result = pred.detach().cpu().numpy()
    return result


def table_structure(pil_img, imgsz=640):
    image = PIL_to_cv(pil_img)
    pred = structure_model(image, size=imgsz)
    pred = pred.xywhn[0]
    result = pred.detach().cpu().numpy()
    return result


def crop_image(pil_img, detection_result):
    crop_images = []
    image = PIL_to_cv(pil_img)
    width = image.shape[1]
    height = image.shape[0]
    # print(width, height)
    for idx, result in enumerate(detection_result):
        class_id = int(result[5])
        score = float(result[4])
        min_x = result[0]
        min_y = result[1]
        w = result[2]
        h = result[3]

        if score < detection_class_thresholds[detection_class_names[class_id]]:
            continue

        x1 = int((min_x - w / 2) * width)
        y1 = int((min_y - h / 2) * height)
        x2 = int((min_x + w / 2) * width)
        y2 = int((min_y + h / 2) * height)
        # print(x1, y1, x2, y2)

        padding_x = max(int(0.02 * width), 30)
        padding_y = max(int(0.02 * height), 30)

        x1_pad = max(0, x1 - padding_x)
        y1_pad = max(0, y1 - padding_y)
        x2_pad = min(width, x2 + padding_x)
        y2_pad = min(height, y2 + padding_y)

        crop_image = image[y1_pad:y2_pad, x1_pad:x2_pad, :]
        crop_image = cv_to_PIL(crop_image)
        if detection_class_names[class_id] == 'table rotated':
            crop_image = crop_image.rotate(270, expand=True)

        crop_images.append(crop_image)

        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        label = f'{detection_class_names[class_id]} {score:.2f}'

        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        fontScale = lw / 3
        thickness = max(lw - 1, 1)
        w_label, h_label = cv2.getTextSize(label, 0, fontScale=fontScale, thickness=thickness)[0]
        cv2.rectangle(image, (x1, y1), (x1 + w_label, y1 - h_label - 3), (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)

    return crop_images, cv_to_PIL(image)


def ocr(pil_img):
    image = PIL_to_cv(pil_img)
    result = ocr_instance.ocr(image)
    ocr_res = []

    for ps, (text, score) in result[0]:
        x1 = min(p[0] for p in ps)
        y1 = min(p[1] for p in ps)
        x2 = max(p[0] for p in ps)
        y2 = max(p[1] for p in ps)
        word_info = {
            'bbox': [x1, y1, x2, y2],
            'text': text
        }
        ocr_res.append(word_info)

    return ocr_res


def convert_stucture(page_tokens, pil_img, structure_result):
    image = PIL_to_cv(pil_img)

    width = image.shape[1]
    height = image.shape[0]
    # print(width, height)

    bboxes = []
    scores = []
    labels = []
    for idx, result in enumerate(structure_result):
        class_id = int(result[5])
        score = float(result[4])
        min_x = result[0]
        min_y = result[1]
        w = result[2]
        h = result[3]

        x1 = int((min_x - w / 2) * width)
        y1 = int((min_y - h / 2) * height)
        x2 = int((min_x + w / 2) * width)
        y2 = int((min_y + h / 2) * height)
        # print(x1, y1, x2, y2)

        bboxes.append([x1, y1, x2, y2])
        scores.append(score)
        labels.append(class_id)

    table_objects = []
    for bbox, score, label in zip(bboxes, scores, labels):
        table_objects.append({'bbox': bbox, 'score': score, 'label': label})
    # print('table_objects:', table_objects)

    table = {'objects': table_objects, 'page_num': 0}

    table_class_objects = [obj for obj in table_objects if obj['label'] == structure_class_map['table']]
    if len(table_class_objects) > 1:
        table_class_objects = sorted(table_class_objects, key=lambda x: x['score'], reverse=True)
    try:
        table_bbox = list(table_class_objects[0]['bbox'])
    except:
        table_bbox = (0, 0, 1000, 1000)
    # print('table_class_objects:', table_class_objects)
    # print('table_bbox:', table_bbox)

    tmp = Rect(table_bbox)
    for obj in table_objects:
        if structure_class_names[obj['label']] in ('table column', 'table row'):
            if postprocess.iob(obj['bbox'], table_bbox) >= 0.001:
                tmp.include_rect(obj['bbox'])
    table_bbox = (tmp[0], tmp[1], tmp[2], tmp[3])

    tokens_in_table = [token for token in page_tokens if postprocess.iob(token['bbox'], table_bbox) >= 0.001]
    # print('tokens_in_table:', tokens_in_table)

    table_structures, cells, confidence_score = postprocess.objects_to_cells(table, table_objects, tokens_in_table, structure_class_names, structure_class_thresholds)

    return table_structures, cells, confidence_score


def visualize_image(pil_img):
    plt.imshow(pil_img, interpolation='lanczos')
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, bbox_inches='tight', dpi=150)
    plt.close()
    return PIL.Image.open(img_buf)


def visualize_ocr(pil_img, ocr_result):
    plt.imshow(pil_img, interpolation='lanczos')
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()

    for idx, result in enumerate(ocr_result):
        bbox = result['bbox']
        text = result['text']
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=2, edgecolor='red', facecolor='none', linestyle='-')
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], text, horizontalalignment='left', verticalalignment='bottom', color='blue', fontsize=7)

    plt.xticks([], [])
    plt.yticks([], [])

    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, bbox_inches='tight', dpi=150)
    plt.close()

    return PIL.Image.open(img_buf)


def get_bbox_decorations(data_type, label):
    if label == 0:
        if data_type == 'detection':
            return 'brown', 0.05, 3, '//'
        else:
            return 'brown', 0, 3, None
    elif label == 1:
        return 'red', 0.15, 2, None
    elif label == 2:
        return 'blue', 0.15, 2, None
    elif label == 3:
        return 'magenta', 0.2, 3, '//'
    elif label == 4:
        return 'cyan', 0.2, 4, '//'
    elif label == 5:
        return 'green', 0.2, 4, '\\\\'

    return 'gray', 0, 0, None


def visualize_structure(pil_img, structure_result):
    image = PIL_to_cv(pil_img)
    width = image.shape[1]
    height = image.shape[0]
    # print(width, height)

    plt.imshow(pil_img, interpolation='lanczos')
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()

    for idx, result in enumerate(structure_result):
        class_id = int(result[5])
        score = float(result[4])
        min_x = result[0]
        min_y = result[1]
        w = result[2]
        h = result[3]

        if score < structure_class_thresholds[structure_class_names[class_id]]:
            continue

        x1 = int((min_x - w / 2) * width)
        y1 = int((min_y - h / 2) * height)
        x2 = int((min_x + w / 2) * width)
        y2 = int((min_y + h / 2) * height)
        # print(x1, y1, x2, y2)
        bbox = [x1, y1, x2, y2]

        color, alpha, linewidth, hatch = get_bbox_decorations('recognition', class_id)
        # Fill
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1],
                                    linewidth=linewidth, alpha=alpha,
                                    edgecolor='none',facecolor=color,
                                    linestyle=None)
        ax.add_patch(rect)
        # Hatch
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1],
                                    linewidth=1, alpha=0.4,
                                    edgecolor=color, facecolor='none',
                                    linestyle='--',hatch=hatch)
        ax.add_patch(rect)
        # Edge
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1],
                                    linewidth=linewidth,
                                    edgecolor=color, facecolor='none',
                                    linestyle='--')
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = []
    for class_name in structure_class_names[:-1]:
        color, alpha, linewidth, hatch = get_bbox_decorations('recognition', structure_class_map[class_name])
        legend_elements.append(
            Patch(facecolor='none', edgecolor=color, linestyle='--', label=class_name, hatch=hatch)
        )

    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
                    fontsize=10, ncol=3)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, bbox_inches='tight', dpi=150)
    plt.close()

    return PIL.Image.open(img_buf)


def visualize_cells(pil_img, cells):
    plt.imshow(pil_img, interpolation='lanczos')
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()

    for cell in cells:
        bbox = cell['bbox']

        if cell['header']:
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        elif cell['subheader']:
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        else:
            facecolor = (0.3, 0.74, 0.8)
            edgecolor = (0.3, 0.7, 0.6)
            alpha = 0.3
            linewidth = 2
            hatch='\\\\\\\\\\\\'

        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor='none',facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(0.3, 0.74, 0.8), edgecolor=(0.3, 0.7, 0.6),
                                label='Data cell', hatch='\\\\\\\\\\\\', alpha=0.3),
                        Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                                label='Column header cell', hatch='//////', alpha=0.3),
                        Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                                label='Projected row header cell', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
                    fontsize=10, ncol=3)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, bbox_inches='tight', dpi=150)
    plt.close()

    return PIL.Image.open(img_buf)


def extract_text_from_cells(cells, sep=' '):
    for cell in cells:
        spans = cell['spans']
        text = ''
        for span in spans:
            if 'text' in span:
                text += span['text'] + sep
        cell['cell_text'] = text
    return cells


def cells_to_csv(cells):
    if len(cells) > 0:
        num_columns = max([max(cell['column_nums']) for cell in cells]) + 1
        num_rows = max([max(cell['row_nums']) for cell in cells]) + 1
    else:
        return

    header_cells = [cell for cell in cells if cell['header']]
    if len(header_cells) > 0:
        max_header_row = max([max(cell['row_nums']) for cell in header_cells])
    else:
        max_header_row = -1

    table_array = np.empty([num_rows, num_columns], dtype='object')
    if len(cells) > 0:
        for cell in cells:
            for row_num in cell['row_nums']:
                for column_num in cell['column_nums']:
                    table_array[row_num, column_num] = cell['cell_text']

    header = table_array[:max_header_row+1,:]
    flattened_header = []
    for col in header.transpose():
        flattened_header.append(' | '.join(OrderedDict.fromkeys(col)))
    df = pd.DataFrame(table_array[max_header_row+1:,:], index=None, columns=flattened_header)

    return df, df.to_csv(index=None)


def cells_to_html(cells):
    cells = sorted(cells, key=lambda k: min(k['column_nums']))
    cells = sorted(cells, key=lambda k: min(k['row_nums']))

    table = ET.Element('table')
    current_row = -1

    for cell in cells:
        this_row = min(cell['row_nums'])

        attrib = {}
        colspan = len(cell['column_nums'])
        if colspan > 1:
            attrib['colspan'] = str(colspan)
        rowspan = len(cell['row_nums'])
        if rowspan > 1:
            attrib['rowspan'] = str(rowspan)
        if this_row > current_row:
            current_row = this_row
            if cell['header']:
                cell_tag = 'th'
                row = ET.SubElement(table, 'tr')
            else:
                cell_tag = 'td'
                row = ET.SubElement(table, 'tr')
        tcell = ET.SubElement(row, cell_tag, attrib=attrib)
        tcell.text = cell['cell_text']

    return str(ET.tostring(table, encoding='unicode', short_empty_elements=False))


# def cells_to_html(cells):
#     for cell in cells:
#         cell['column_nums'].sort()
#         cell['row_nums'].sort()
#     n_cols = max(cell['column_nums'][-1] for cell in cells) + 1
#     n_rows = max(cell['row_nums'][-1] for cell in cells) + 1
#     html_code = ''
#     for r in range(n_rows):
#         r_cells = [cell for cell in cells if cell['row_nums'][0] == r]
#         r_cells.sort(key=lambda x: x['column_nums'][0])
#         r_html = ''
#         for cell in r_cells:
#             rowspan = cell['row_nums'][-1] - cell['row_nums'][0] + 1
#             colspan = cell['column_nums'][-1] - cell['column_nums'][0] + 1
#             r_html += f'<td rowspan='{rowspan}' colspan='{colspan}'>{escape(cell['text'])}</td>'
#         html_code += f'<tr>{r_html}</tr>'
#     html_code = '''<html>
#                    <head>
#                    <meta charset='UTF-8'>
#                    <style>
#                    table, th, td {
#                      border: 1px solid black;
#                      font-size: 10px;
#                    }
#                    </style>
#                    </head>
#                    <body>
#                    <table frame='hsides' rules='groups' width='100%%'>
#                      %s
#                    </table>
#                    </body>
#                    </html>''' % html_code
#     soup = bs(html_code)
#     html_code = soup.prettify()
#     return html_code


def cells_to_excel(cells, file_path):

    def int2xlsx(i):
        if i < 26:
            return chr(i + 65)
        return f'{chr(i // 26 + 64)}{chr(i % 26 + 65)}'

    cells = sorted(cells, key=lambda k: min(k['column_nums']))
    cells = sorted(cells, key=lambda k: min(k['row_nums']))

    workbook = xlsxwriter.Workbook(file_path)

    cell_format = workbook.add_format(
        {'align': 'center', 'valign': 'vcenter'}
    )

    worksheet = workbook.add_worksheet(name='Table')

    table_start_index = 0

    for cell in cells:
        start_row = min(cell['row_nums'])
        end_row = max(cell['row_nums'])
        start_col = min(cell['column_nums'])
        end_col = max(cell['column_nums'])
        if start_row == end_row and start_col == end_col:
            worksheet.write(
                table_start_index + start_row,
                start_col,
                cell['cell_text'],
                cell_format,
            )
        else:
            if start_col == end_col and start_row == end_row:
                excel_index = f'{int2xlsx(table_start_index + start_col)}{table_start_index + start_row + 1}'
            else:
                excel_index = f'{int2xlsx(table_start_index + start_col)}{table_start_index + start_row + 1}:{int2xlsx(table_start_index + end_col)}{table_start_index + end_row + 1}'
            worksheet.merge_range(
                excel_index, cell['cell_text'], cell_format
            )

    workbook.close()


def main():

    st.title('Table Extraction Demo')

    filename = st.file_uploader('Upload image', type=['png', 'jpeg', 'jpg'])

    if st.button('Analyze image'):

        if filename is None:
            st.write('Please upload an image')

        else:
            tabs = st.tabs(
                ['Table Detection', 'Table Structure Recognition', 'Extracted Table(s)']
            )

            print(filename)
            pil_img = PIL.Image.open(filename)

            detection_result = table_detection(pil_img)
            crop_images, vis_det_img = crop_image(pil_img, detection_result)

            all_cells = []

            with tabs[0]:
                st.header('Table Detection')
                st.image(vis_det_img)

            with tabs[1]:
                st.header('Table Structure Recognition')

                str_cols = st.columns(4)
                str_cols[0].subheader('Table image')
                str_cols[1].subheader('OCR result')
                str_cols[2].subheader('Structure result')
                str_cols[3].subheader('Cells result')

                for idx, img in enumerate(crop_images):
                    str_cols = st.columns(4)

                    vis_img = visualize_image(img)
                    str_cols[0].image(vis_img)

                    ocr_result = ocr(img)
                    vis_ocr_img = visualize_ocr(img, ocr_result)
                    str_cols[1].image(vis_ocr_img)

                    structure_result = table_structure(img)
                    vis_str_img = visualize_structure(img, structure_result)
                    str_cols[2].image(vis_str_img)

                    table_structures, cells, confidence_score = convert_stucture(ocr_result, img, structure_result)
                    cells = extract_text_from_cells(cells)
                    vis_cells_img = visualize_cells(img, cells)
                    str_cols[3].image(vis_cells_img)

                    all_cells.append(cells)

                    #df, csv_result = cells_to_csv(cells)
                    #print(df)

            with tabs[2]:
                st.header('Extracted Table(s)')
                for idx, col in enumerate(st.columns(len(all_cells))):
                    with col:
                        if len(all_cells) > 1:
                            st.header(f'Table {idx + 1}')

                        with TemporaryDirectory() as temp_dir_path:
                            df = None
                            xlsx_path = os.path.join(temp_dir_path, f'debug_{idx}.xlsx')
                            cells_to_excel(all_cells[idx], xlsx_path)
                            with open(xlsx_path, 'rb') as ref:
                                df = pd.read_excel(ref)
                                st.dataframe(df)
                                st.download_button(
                                    'Download Excel File',
                                    ref,
                                    file_name=f'output_{idx}.xlsx',
                                )

                for idx, cells in enumerate(all_cells):
                    html_result = cells_to_html(cells)
                    st.subheader(f'HTML Table {idx + 1}')
                    st.markdown(html_result, unsafe_allow_html=True)


if __name__ == '__main__':
    main()