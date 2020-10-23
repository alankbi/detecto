import torch

from detecto.core import *
from detecto.utils import read_image
from .helpers import get_dataset, get_image, get_model, empty_predictor
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Test that the dataset returns the correct things and properly
# applies the default or given transforms
def test_dataset():
    # Test the format of the values returned by indexing the dataset
    dataset = get_dataset()
    assert len(dataset) == 1  # there is only one image in the dataset (label.xml)
    assert isinstance(dataset[0][0], torch.Tensor)
    assert isinstance(dataset[0][1], dict)
    assert dataset[0][0].shape == (3, 1080, 1720)
    assert 'boxes' in dataset[0][1] and 'labels' in dataset[0][1]
    assert dataset[0][1]['boxes'].shape == (2, 4)
    assert dataset[0][1]['labels'] == ['start_tick', 'start_gate']

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(108),
        transforms.RandomHorizontalFlip(1),
        transforms.ToTensor()
    ])

    # Test that the transforms are properly applied
    dataset = get_dataset(transform=transform)
    assert dataset[0][0].shape == (3, 108, 172)
    assert torch.all(dataset[0][1]['boxes'][1] == torch.tensor([6, 41, 171, 107]))

    # Test works when given an XML folder
    path = os.path.dirname(__file__)
    input_folder = os.path.join(path, 'static')

    dataset = Dataset(input_folder, input_folder)
    assert len(dataset) == 1
    assert dataset[0][0].shape == (3, 1080, 1720)
    assert 'boxes' in dataset[0][1] and 'labels' in dataset[0][1]

    dataset = Dataset(input_folder)
    assert len(dataset) == 1
    assert dataset[0][0].shape == (3, 1080, 1720)
    assert 'boxes' in dataset[0][1] and 'labels' in dataset[0][1]


# Ensure that the collate function of the DataLoader properly
# converts a list of tuples into a tuple of lists
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


# Test that the dataloader correctly loops through every element
# in the dataset and returns them in the right format
def test_dataloader():
    dataset = get_dataset()
    loader = DataLoader(dataset, batch_size=2)

    iterations = 0
    for data in loader:
        iterations += 1

        assert isinstance(data, tuple)
        assert len(data) == 2  # data[0] = image tensor, data[1] = targets dictionary
        assert isinstance(data[0], list)
        assert len(data[0]) == 1  # only one image in data[0] since label.xml contains one image only.

        assert isinstance(data[0][0], torch.Tensor)
        assert 'boxes' in data[1][0] and 'labels' in data[1][0]

    assert 'boxes' in dataset[0][1] and 'labels' in dataset[0][1]
    assert iterations == 1


# Ensure that the model's internal parameters are properly set
def test_model_internal():
    model = get_model()

    assert model._device == torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    box_predictor = model._model.roi_heads.box_predictor
    assert isinstance(box_predictor, FastRCNNPredictor)
    assert box_predictor.cls_score.out_features == 4

    assert model._classes == ['__background__', 'test1', 'test2', 'test3']
    assert model._int_mapping['test1'] == 1

    # _int_mapping should give the right index of each class
    for k in model._int_mapping:
        assert model._classes[model._int_mapping[k]] == k


# def test_model_default():
#     path = os.path.dirname(__file__)
#     file = os.path.join(path, 'static/apple_orange.jpg')
#
#     model = Model()
#     preds = model.predict_top(read_image(file))
#
#     assert len(preds[0]) >= 2
#     assert 'orange' in preds[0] and 'apple' in preds[0]
#     assert sum(preds[2]) / len(preds[2]) > 0.50


# Ensure that fitting the model increases accuracy and returns the losses
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

    losses = model.fit(loader, val_dataset=loader, epochs=1)

    # Average loss during training should be lower than initial loss
    assert len(losses) == 1
    assert losses[0] < initial_loss

    # Should not return anything if not validation losses are produced
    losses = model.fit(loader, loader, epochs=0)
    assert losses is None

    # Works when given datasets
    losses = model.fit(dataset, val_dataset=dataset, epochs=1)
    assert len(losses) == 1


# Test both the predict and predict_top methods with both single
# images and lists of images to predict on
# TODO: test applying transforms on images to predict
def test_model_predict():
    classes = ['start_tick', 'start_gate']
    path = os.path.dirname(__file__)
    file = os.path.join(path, 'static/model.pth')

    # Load in a pre-fitted model so it can actually make predictions
    model = Model.load(file, classes)
    image = get_image()

    # Test predict method on a single image
    pred = model.predict(image)
    assert isinstance(pred, tuple)
    assert set(pred[0]) == set(classes)
    assert isinstance(pred[1][0], torch.Tensor) and pred[1][0].shape[0] == 4
    assert pred[2][0] > 0.5

    # Test predict method on a list of images
    preds = model.predict([image])
    assert len(preds) == 1
    assert preds[0][0] == pred[0]
    assert torch.all(preds[0][1] == pred[1])
    assert torch.all(preds[0][2] == pred[2])

    # Test predict_top method on a single image
    top_pred = model.predict_top(image)
    assert isinstance(top_pred, tuple)
    assert len(top_pred[0]) == 2
    assert isinstance(top_pred[1], torch.Tensor)
    assert top_pred[1].shape[0] == len(top_pred[0]) and top_pred[1].shape[1] == 4
    assert top_pred[2][0] > 0.5

    # Test predict_top method on a list of images
    top_preds = model.predict_top([image])
    assert len(top_preds) == 1
    assert set(top_preds[0][0]) == set(top_pred[0])
    assert torch.all(top_preds[0][1][0] == top_pred[1][0]) or \
        torch.all(top_preds[0][1][0] == top_pred[1][1])
    assert top_preds[0][2][0] == top_pred[2][0] or \
        top_preds[0][2][0] == top_pred[2][1]

    # Test return values when no predictions are made
    model._model.forward = empty_predictor
    preds = model.predict([image])
    assert len(preds) == 1
    assert preds[0][0] == [] and preds[0][1].nelement() == 0 and preds[0][2].nelement() == 0

    preds = model.predict_top([image])
    assert len(preds) == 1
    assert preds[0][0] == [] and preds[0][1].nelement() == 0 and preds[0][2].nelement() == 0


# Test that save, load, and get_internal_model all work properly
def test_model_helpers():
    path = os.path.dirname(__file__)
    file = os.path.join(path, 'static/saved_model.pth')

    model = get_model()

    model.save(file)
    model = Model.load(file, ['test1', 'test2', 'test3'])

    assert model._model is not None
    assert model.get_internal_model() is model._model

    os.remove(file)
