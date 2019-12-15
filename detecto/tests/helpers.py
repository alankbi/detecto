import os

from detecto.core import Model
from skimage import io


def get_image():
    path = os.path.dirname(__file__)
    file = 'static/image.jpg'
    return io.imread(os.path.join(path, file))


def get_model():
    return Model(['test1', 'test2', 'test3'])
