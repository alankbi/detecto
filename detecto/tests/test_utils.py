import os
import pandas as pd
import torch

from detecto.utils import *
from skimage import io


def get_image():
    path = os.path.dirname(__file__)
    file = "static/image.jpg"
    return io.imread(os.path.join(path, file))


def test_filter_top_predictions():
    labels = ['test1', 'test2', 'test1', 'test2', 'test2']
    boxes = torch.ones(5, 4)
    scores = [5, 4, 3, 2, 1]
    for i in range(5):
        boxes[i] *= i

    preds = filter_top_predictions(labels, boxes, scores)

    assert len(preds) == 2
    # Correct labels
    assert {preds[0][0], preds[1][0]} == {'test1', 'test2'}

    # Correct box coordinates
    assert {preds[0][1][0].item(), preds[1][1][0].item()} == {0, 1}

    # Correct scores
    assert {preds[0][2], preds[1][2]} == {5, 4}


def test_normalize_functions():
    transform = normalize_transform()
    image = transforms.ToTensor()(get_image())

    normalized_img = transform(image)
    reversed_img = reverse_normalize(normalized_img)

    assert (image - reversed_img).max() < 0.05


def test_xml_to_csv():
    path = os.path.dirname(__file__)
    input_folder = os.path.join(path, 'static')
    output_path = os.path.join(path, 'static/labels.csv')

    xml_to_csv(input_folder, output_path)
    csv = pd.read_csv(output_path)

    assert len(csv) == 2
    assert csv.loc[0, 'filename'] == 'frame199.jpg'
    assert csv.loc[1, 'class'] == 'start_gate'
    assert csv.loc[0, 'width'] == 1720
    assert csv.loc[1, 'height'] == 1080
    assert csv.loc[0, 'ymax'] == 784
    assert csv.loc[1, 'xmin'] == 1

