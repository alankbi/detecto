from .helpers import get_image
from requests import get
import logging
import os

MODEL_URL = "https://www.dropbox.com/s/kjalrfs3la97du8/model.pth?dl=1"
LOGGER = logging.getLogger(__name__)


def _download_model_file():
    path = os.path.dirname(__file__)
    model_path = os.path.join(path, 'static/model.pth')

    if os.path.isfile(model_path):
        LOGGER.info("Model file already exists. Continuing.")
        return

    LOGGER.info("\nDownloading model file......")

    with open(model_path, "wb") as model_file:
        response = get(MODEL_URL)
        model_file.write(response.content)

    LOGGER.info("Done")


_download_model_file()