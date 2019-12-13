from torchvision import transforms
from glob import glob
import pandas as pd
import xml.etree.ElementTree as ET


def filter_top_predictions(labels, boxes, scores):
    preds = []
    # Loop through each unique label
    for label in set(labels):
        # Get first index of label, which is also its highest scoring occurrence
        index = labels.index(label)
        preds.append((label, boxes[index], scores[index]))
    return preds


def normalize_transform():
    # Default for PyTorch's pre-trained models
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def reverse_normalize(image):
    reverse = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
    return reverse(image)


def xml_to_csv(path, output_path):
    xml_list = []
    for xml_file in glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text
        width = int(root.find('size')[0].text)
        height = int(root.find('size')[1].text)

        for member in root.findall('object'):
            box = member.find('bndbox')
            label = member.find('name').text
            row = (filename, width, height, label, int(box[0].text),
                   int(box[1].text), int(box[2].text), int(box[3].text))
            xml_list.append(row)

    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_names)
    xml_df.to_csv(output_path, index=None)
