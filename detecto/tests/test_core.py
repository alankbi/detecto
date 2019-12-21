import torch

from detecto.core import *
from .helpers import get_dataset, get_model
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def test_dataset():
    dataset = get_dataset()
    assert len(dataset) == 2
    assert isinstance(dataset[0][0], torch.Tensor)
    assert isinstance(dataset[0][1], dict)
    assert dataset[0][0].shape == (3, 1080, 1720)
    assert 'boxes' in dataset[0][1] and 'labels' in dataset[0][1]
    assert dataset[0][1]['boxes'].shape == (1, 4)
    assert dataset[0][1]['labels'] == 'start_tick'

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(108),
        transforms.RandomHorizontalFlip(1),
        transforms.ToTensor()
    ])
    dataset = get_dataset(transform=transform)
    assert dataset[1][0].shape == (3, 108, 172)
    assert torch.all(dataset[1][1]['boxes'][0] == torch.tensor([6, 41, 171, 107]))


def test_collate_fn():
    test_input = [(1, 10), (2, 20), (3, 30)]
    test_output = DataLoader.collate_data(test_input)

    assert isinstance(test_output, tuple)
    assert len(test_output) == 2
    assert isinstance(test_output[0], list)
    assert len(test_output[0]) == 3

    assert test_output[0][0] == 1
    assert test_output[0][2] == 3
    assert test_output[1][0] == 10
    assert test_output[1][2] == 30


def test_dataloader():
    dataset = get_dataset()
    loader = DataLoader(dataset, batch_size=2)

    iterations = 0
    for data in loader:
        iterations += 1

        assert isinstance(data, tuple)
        assert len(data) == 2
        assert isinstance(data[0], list)
        assert len(data[0]) == 2

        assert isinstance(data[0][0], torch.Tensor)
        assert isinstance(data[0][1], torch.Tensor)
        assert 'boxes' in dataset[0][1] and 'labels' in dataset[1][1]

    assert iterations == 1


def test_model_internal():
    model = get_model()

    assert model._device == torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    box_predictor = model._model.roi_heads.box_predictor
    assert isinstance(box_predictor, FastRCNNPredictor)
    assert box_predictor.cls_score.out_features == 4

    assert model._classes == ['__background__', 'test1', 'test2', 'test3']
    assert model._int_mapping['test1'] == 1

    for k in model._int_mapping:
        assert model._classes[model._int_mapping[k]] == k


def test_model_fit():
    pass


def test_model_predict():
    pass


def test_model_save_load():
    pass


def test_model_helpers():
    pass
