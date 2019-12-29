import pandas as pd
from skimage import io
import torch
import xml.etree.ElementTree as ET

from torchvision import transforms
from glob import glob


def filter_top_predictions(labels, boxes, scores):
    """Filters out the top scoring predictions of each class from the
    given data. Note: passing the predictions from
    :meth:`detecto.Model.predict` to this function produces the same
    results as a direct call to :meth:`detecto.Model.predict_top`.

    :param labels: A list containing the string labels.
    :type labels: list
    :param boxes: A tensor of size [N, 4] containing the N box coordinates.
    :type boxes: torch.Tensor
    :param scores: A tensor containing the score for each prediction.
    :type scores: torch.Tensor
    :return: Returns a list of size K, where K is the number of uniquely
        predicted classes in ``labels``. Each element in the list is a tuple
        containing the label, a tensor of size 4 containing the box
        coordinates, and the score for that prediction.
    :rtype: list of tuple

    **Example**::

        >>> from detecto.core import Model
        >>> from detecto.utils import read_image, filter_top_predictions
        >>>
        >>> model = Model.load('model_weights.pth', ['label1', 'label2'])
        >>> image = read_image('image.jpg')
        >>> labels, boxes, scores = model.predict(image)
        >>> top_preds = filter_top_predictions(labels, boxes, scores)
        >>> top_preds
        [('label2', tensor([859.4128, 415.1042, 904.5725, 659.7365]),
        tensor(0.8788)), ('label1', tensor([ 281.3972,  463.2599, 1303.1023,
         969.5024]), tensor(0.9040))]
    """

    preds = []
    # Loop through each unique label
    for label in set(labels):
        # Get first index of label, which is also its highest scoring occurrence
        index = labels.index(label)
        preds.append((label, boxes[index], scores[index]))
    return preds


def default_transforms():
    return transforms.Compose([transforms.ToTensor(), normalize_transform()])


def normalize_transform():
    # Default for PyTorch's pre-trained models
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def read_image(path):
    return io.imread(path)


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


def _is_iterable(variable):
    return isinstance(variable, list) or isinstance(variable, tuple)
