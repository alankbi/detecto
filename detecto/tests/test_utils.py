import os
import pandas as pd
import pytest
import torch
import torchvision

from .helpers import get_image
from detecto.utils import *
from detecto.utils import _is_iterable


def test_filter_top_predictions():
    labels = ['test1', 'test2', 'test1', 'test2', 'test2']
    boxes = torch.ones(5, 4)
    scores = torch.tensor([5., 4, 3, 2, 1])
    for i in range(5):
        boxes[i] *= i

    preds = filter_top_predictions(labels, boxes, scores)

    assert isinstance(preds, tuple) and len(preds) == 3
    # Correct labels
    assert len(preds[0]) == 2 and set(preds[0]) == {'test1', 'test2'}

    # Correct box coordinates
    assert {preds[1][0][0].item(), preds[1][1][0].item()} == {0, 1}

    # Correct scores
    assert {preds[2][0].item(), preds[2][1].item()} == {5, 4}


def test_default_transforms():
    transforms = default_transforms()

    assert isinstance(transforms.transforms[0], torchvision.transforms.ToTensor)
    assert isinstance(transforms.transforms[1], torchvision.transforms.Normalize)
    assert transforms.transforms[1].mean == normalize_transform().mean
    assert transforms.transforms[1].std == normalize_transform().std


def test_normalize_functions():
    transform = normalize_transform()
    image = transforms.ToTensor()(get_image())

    normalized_img = transform(image)
    reversed_img = reverse_normalize(normalized_img)

    # Normalized image that's then reversed should be close to original
    assert (image - reversed_img).max() < 0.05


def test_read_image():
    path = os.path.dirname(__file__)
    file = 'static/image.jpg'
    image_path = os.path.join(path, file)

    image = get_image()

    assert (read_image(image_path) == image).all()


def test_read_image_fails_with_image_not_found():
    image_path = 'foo/bar'

    with pytest.raises(ValueError) as e:
        read_image(image_path)

    assert 'Could not read image foo/bar' == str(e.value)


def test_read_image_fails_with_cv_error():
    image_path = 'static/demo.gif'

    with pytest.raises(ValueError) as e:
        read_image(image_path)

    assert 'Could not read image static/demo.gif' in str(e.value)


def test_split_video():
    path = os.path.dirname(__file__)
    input_video = os.path.join(path, 'static/input_video.mp4')
    output_path = os.path.join(path, 'static/split_frames')

    os.mkdir(output_path)

    split_video(input_video, output_path, prefix='testing', step_size=-1)

    files = [f for f in os.listdir(output_path)]

    assert len(files) == 1
    assert files[0] == 'testing0.jpg'

    os.remove(os.path.join(output_path, 'testing0.jpg'))
    os.rmdir(output_path)


def test_xml_to_csv():
    path = os.path.dirname(__file__)
    input_folder = os.path.join(path, 'static')
    output_path = os.path.join(path, 'static/labels.csv')

    df = xml_to_csv(input_folder, output_path)
    csv = pd.read_csv(output_path)

    assert len(csv) == 2
    assert len(df) == 2
    assert csv.loc[0, 'filename'] == 'image.jpg'
    assert df.loc[1, 'class'] == 'start_gate'
    assert csv.loc[0, 'width'] == 1720
    assert df.loc[1, 'height'] == 1080
    assert csv.loc[0, 'ymax'] == 784
    assert df.loc[1, 'xmin'] == 1

    os.remove(output_path)

    xml_to_csv(input_folder)
    assert not os.path.isfile(output_path)


def test__is_iterable():
    test1 = [1, 2]
    test2 = (3, 4)
    test3 = torch.ones(2)
    test4 = 5

    assert _is_iterable(test1)
    assert _is_iterable(test2)
    assert not _is_iterable(test3)
    assert not _is_iterable(test4)
