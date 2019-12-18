from detecto.core import *


def test_dataset():
    pass


def test_collate_fn():
    test_input = [(1, 10), (2, 20), (3, 30)]
    test_output = DataLoader.collate_data(test_input)

    assert isinstance(test_output, tuple)
    assert isinstance(test_output[0], list)
    assert len(test_output) == 2
    assert len(test_output[0]) == 3

    assert test_output[0][0] == 1
    assert test_output[0][2] == 3
    assert test_output[1][0] == 10
    assert test_output[1][2] == 30


def test_dataloader():
    pass


def test_model():
    pass
