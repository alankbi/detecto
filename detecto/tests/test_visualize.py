import os

from .helpers import get_image, get_model
from detecto.visualize import *


def test_detect_video():
    path = os.path.dirname(__file__)
    input_video = os.path.join(path, 'static/input_video.mp4')
    output_video = os.path.join(path, 'static/output_video.avi')

    model = get_model()
    detect_video(model, input_video, output_video)

    assert os.path.isfile(output_video)
    os.remove(output_video)


def test_plot_prediction_grid():
    model = get_model()
    try:
        plot_prediction_grid(model, [3, 4], (3, 5))
        assert False  # Above should throw a value error
    except ValueError:
        pass
    except Exception:
        assert False  # An error occurred

    image = get_image()
    plot_prediction_grid(model, [image], 1, show=False)  # Shouldn't throw an error


def test_show_labeled_image():
    image = get_image()

    # Shouldn't throw any errors
    show_labeled_image(image, torch.ones(4), show=False)
    show_labeled_image(image, torch.ones(10, 4), show=False)
