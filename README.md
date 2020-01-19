[![CircleCI](https://circleci.com/gh/alankbi/detecto/tree/master.svg?style=svg)](https://circleci.com/gh/alankbi/detecto/tree/master)
[![Documentation Status](https://readthedocs.org/projects/detecto/badge/?version=latest)](https://detecto.readthedocs.io/en/latest/?badge=latest)

# Detecto

Detecto is a Python package for quick and easy object detection. Below are just a few of the features available:

* Train models on custom datasets
* Get all or top predictions on an image
* Run object detection on videos
* Save and load models from files

Detecto is built on top of PyTorch, meaning models trained with Detecto can easily be extracted and used with PyTorch code. 

![Video demo of Detecto](demo.gif)

## Usage and Docs

To install Detecto using pip, run the following command:

`pip install detecto`

After installing Detecto, you can train a machine learning model on a custom dataset and run object detection on a video with under ten lines of code:

```python
from detecto.core import Model, Dataset, DataLoader
from detecto.utils import xml_to_csv
from detecto.visualize import detect_video

xml_to_csv('xml_labels/', 'labels.csv')
dataset = Dataset('labels.csv', 'images/')
loader = DataLoader(dataset)

model = Model(['dog', 'cat', 'rabbit'])
model.fit(loader)

detect_video(model, 'input_video.mp4', 'output_video.avi')
```

Visit the [docs](https://detecto.readthedocs.io/) for a full guide, including a [quickstart](https://detecto.readthedocs.io/en/latest/usage/quickstart.html) tutorial.

Alternatively, check out the [demo on Colab](https://colab.research.google.com/drive/1ISaTV5F-7b4i2QqtjTa7ToDPQ2k8qEe0).  

## Contributing

All issues and pull requests are welcome! To run the code locally, first fork the repository and then run the following commands on your computer: 

```bash
git clone https://github.com/<your-username>/detecto.git
cd detecto
# Recommended to create a virtual environment before the next step
pip3 install -r requirements.txt
```

When adding code, be sure to write unit tests and docstrings where necessary. 

Tests are located in `detecto/tests` and can be run using pytest:

`python -m pytest`

To generate the documentation locally, run the following commands:

```bash
cd docs
make html
```

The documentation can then be viewed at `docs/_build/html/index.html`.

## Contact

Detecto was created by [Alan Bi](https://www.alanbi.com/). Feel free to reach out on [Twitter](https://twitter.com/alankbi) or through [email](mailto:alan.bi326@gmail.com)!
