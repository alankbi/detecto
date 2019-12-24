import torch

from detecto.core import *
from .helpers import get_dataset, get_image, get_model
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
    model = Model(['start_tick', 'start_gate'])

    dataset = get_dataset()
    loader = DataLoader(dataset, batch_size=1)

    initial_loss = 0
    with torch.no_grad():
        for images, targets in loader:
            model._convert_to_int_labels(targets)
            images, targets = model._to_device(images, targets)
            loss_dict = model._model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            initial_loss += total_loss.item()
    initial_loss /= len(loader.dataset)

    losses = model.fit(loader, val_loader=loader, epochs=2)

    assert len(losses) == 2
    assert sum(losses) / 2 < initial_loss

    losses = model.fit(loader, loader, epochs=0)
    assert losses is None


def test_model_predict():
    classes = ['start_tick', 'start_gate']
    path = os.path.dirname(__file__)
    file = os.path.join(path, 'static/model.pth')

    model = Model.load(file, classes)
    image = get_image()

    # Test predict method
    pred = model.predict(image)
    print('pred')
    print(pred)

    assert isinstance(pred, tuple)
    assert set(pred[0]) == set(classes)
    assert isinstance(pred[1][0], torch.Tensor) and pred[1][0].shape[0] == 4
    assert pred[2][0] > 0.5

    preds = model.predict([image])
    print('preds')
    print(preds)
    assert len(preds) == 1
    assert preds[0][0] == pred[0]
    assert torch.all(preds[0][1] == pred[1])
    assert torch.all(preds[0][2] == pred[2])

    # Test predict_top method
    top_preds = model.predict_top([image])
    print('top preds')
    print(top_preds)
    assert len(top_preds) == 1

    top_pred = top_preds[0]
    assert len(top_pred) == 2 and len(top_pred[0]) == 3
    assert {top_pred[0][0], top_pred[1][0]} == {'start_tick', 'start_gate'}
    assert isinstance(top_pred[0][1], torch.Tensor) and top_pred[0][1].shape[0] == 4
    assert top_pred[0][2] == pred[2][0] or top_pred[1][2] == pred[2][0]

    top_pred = model.predict_top(image)
    print('top pred')
    print(top_pred)
    if top_pred[0][0] == top_preds[0][0][0]:
        assert torch.all(top_pred[0][1] == top_preds[0][0][1])
        assert top_pred[0][2] == top_preds[0][0][2]
        assert torch.all(top_pred[1][1] == top_preds[0][1][1])
        assert top_pred[1][2] == top_preds[0][1][2]
    else:
        assert torch.all(top_pred[0][1] == top_preds[0][1][1])
        assert top_pred[0][2] == top_preds[0][1][2]
        assert torch.all(top_pred[1][1] == top_preds[0][0][1])
        assert top_pred[1][2] == top_preds[0][0][2]


def test_model_helpers():
    path = os.path.dirname(__file__)
    file = os.path.join(path, 'static/saved_model.pth')

    model = get_model()

    model.save(file)
    model = Model.load(file, ['test1', 'test2', 'test3'])

    assert model._model is not None

    os.remove(file)
