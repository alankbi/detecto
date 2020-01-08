import torch

from detecto.core import *
from .helpers import get_dataset, get_image, get_model, empty_predictor
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Test that the dataset returns the correct things and properly
# applies the default or given transforms
def test_dataset():
    # Test the format of the values returned by indexing the dataset
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

    # Test that the transforms are properly applied
    dataset = get_dataset(transform=transform)
    assert dataset[1][0].shape == (3, 108, 172)
    assert torch.all(dataset[1][1]['boxes'][0] == torch.tensor([6, 41, 171, 107]))


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
        assert len(data) == 2
        assert isinstance(data[0], list)
        assert len(data[0]) == 2

        assert isinstance(data[0][0], torch.Tensor)
        assert isinstance(data[0][1], torch.Tensor)
        assert 'boxes' in dataset[0][1] and 'labels' in dataset[1][1]

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

    losses = model.fit(loader, val_loader=loader, epochs=2)

    # Average loss during training should be lower than initial loss
    assert len(losses) == 2
    assert sum(losses) / 2 < initial_loss

    # Should not return anything if not validation losses are produced
    losses = model.fit(loader, loader, epochs=0)
    assert losses is None


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
