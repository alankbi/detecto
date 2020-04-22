import os
import torch

from detecto.core import Model, Dataset
from detecto.utils import xml_to_csv, read_image


def get_dataset(**kwargs):
    path = os.path.dirname(__file__)
    input_folder = os.path.join(path, 'static')
    labels_path = os.path.join(path, 'static/labels.csv')

    xml_to_csv(input_folder, labels_path)
    dataset = Dataset(labels_path, input_folder, **kwargs)
    os.remove(labels_path)

    return dataset


def get_image():
    path = os.path.dirname(__file__)
    file = 'static/image.jpg'
    return read_image(os.path.join(path, file))


def get_model():
    return Model(['test1', 'test2', 'test3'])


def empty_predictor(x):
    return [{'labels': torch.empty(0), 'boxes': torch.empty(0, 4), 'scores': torch.empty(0)}]
