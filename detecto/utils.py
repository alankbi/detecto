import pandas as pd
from skimage import io
import xml.etree.ElementTree as ET

from torchvision import transforms
from glob import glob


def default_transforms():
    """Returns the default, bare-minimum transformations that should be
    applied to images passed to classes in the :mod:`detecto.core` module.

    :return: A torchvision `transforms.Compose
        <https://pytorch.org/docs/stable/torchvision/transforms.html>`_
        object containing a transforms.ToTensor object and the
        transforms.Normalize object returned by
        :func:`detecto.utils.normalize_transform`.
    :rtype: torchvision.transforms.Compose

    **Example**::

        >>> from detecto.core import Dataset
        >>> from detecto.utils import default_transforms

        >>> # Note: if transform=None, the Dataset will automatically
        >>> # apply these default transforms to images
        >>> defaults = default_transforms()
        >>> dataset = Dataset('labels.csv', 'images/', transform=defaults)
    """

    return transforms.Compose([transforms.ToTensor(), normalize_transform()])


def filter_top_predictions(labels, boxes, scores):
    """Filters out the top scoring predictions of each class from the
    given data. Note: passing the predictions from
    :meth:`detecto.core.Model.predict` to this function produces the same
    results as a direct call to :meth:`detecto.core.Model.predict_top`.

    :param labels: A list containing the string labels.
    :type labels: list
    :param boxes: A tensor of size [N, 4] containing the N box coordinates.
    :type boxes: torch.Tensor
    :param scores: A tensor containing the score for each prediction.
    :type scores: torch.Tensor
    :return: Returns a list of size K, where K is the number of uniquely
        predicted classes in ``labels``. Each element in the list is a tuple
        containing the label, a tensor of size 4 containing the box
        coordinates, and the score for that prediction.
    :rtype: list of tuple

    **Example**::

        >>> from detecto.core import Model
        >>> from detecto.utils import read_image, filter_top_predictions

        >>> model = Model.load('model_weights.pth', ['label1', 'label2'])
        >>> image = read_image('image.jpg')
        >>> labels, boxes, scores = model.predict(image)
        >>> top_preds = filter_top_predictions(labels, boxes, scores)
        >>> top_preds
        [('label2', tensor([859.4128, 415.1042, 904.5725, 659.7365]),
        tensor(0.8788)), ('label1', tensor([ 281.3972,  463.2599, 1303.1023,
         969.5024]), tensor(0.9040))]
    """

    preds = []
    # Loop through each unique label
    for label in set(labels):
        # Get first index of label, which is also its highest scoring occurrence
        index = labels.index(label)
        preds.append((label, boxes[index], scores[index]))
    return preds


def normalize_transform():
    """Returns a torchvision `transforms.Normalize
    <https://pytorch.org/docs/stable/torchvision/transforms.html>`_ object
    with default mean and standard deviation values as required by PyTorch's
    pre-trained models.

    :return: A transforms.Normalize object with pre-computed values.
    :rtype: torchvision.transforms.Normalize

    **Example**::

        >>> from detecto.core import Dataset
        >>> from detecto.utils import normalize_transform
        >>> from torchvision import transforms

        >>> # Note: if transform=None, the Dataset will automatically
        >>> # apply these default transforms to images
        >>> defaults = transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     normalize_transform(),
        >>> ])
        >>> dataset = Dataset('labels.csv', 'images/', transform=defaults)
    """

    # Default for PyTorch's pre-trained models
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def read_image(path):
    """Helper function that reads in an image as a
    `NumPy <https://numpy.org/>`_ array. Equivalent to using
    `scikit-image <https://scikit-image.org/>`_'s io.imread function.

    :param path: The path to the image.
    :type path: str
    :return: Image in NumPy array format
    :rtype: ndarray

    **Example**::

        >>> import matplotlib.pyplot as plt
        >>> from detecto.utils import read_image

        >>> image = read_image('image.jpg')
        >>> plt.imshow(image)
        >>> plt.show()
    """

    return io.imread(path)


def reverse_normalize(image):
    """Reverses the normalization applied on an image by the
    :func:`detecto.utils.reverse_normalize` transformation. The image
    must be a `torch.Tensor <https://pytorch.org/docs/stable/tensors.html>`_
    object.

    :param image: A normalized image.
    :type image: torch.Tensor
    :return: The image with the normalization undone.
    :rtype: torch.Tensor


    **Example**::

        >>> import matplotlib.pyplot as plt
        >>> from torchvision import transforms
        >>> from detecto.utils import read_image, \\
        >>>     default_transforms, reverse_normalize

        >>> image = read_image('image.jpg')
        >>> defaults = default_transforms()
        >>> image = defaults(image)

        >>> image = reverse_normalize(image)
        >>> image = transforms.ToPILImage()(image)
        >>> plt.imshow(image)
        >>> plt.show()
    """

    reverse = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
    return reverse(image)


def xml_to_csv(xml_folder, output_file):
    """Converts a folder of XML label files into a single CSV file, which
    can then be used to create a :class:`detecto.core.Dataset` object. Each
    XML file should correspond to an image and contain the image name, image
    size, and the names and bounding boxes of the objects in the image,
    if any. Extraneous data in the XML files will simply be ignored.
    See :download:`here <_static/example.xml>` for an example XML file.
    For an image labeling tool that produces XML files in this format,
    see `LabelImg <https://github.com/tzutalin/labelImg>`_.

    :param xml_folder: The path to the folder containing the XML files.
    :type xml_folder: str
    :param output_file: The name of the output CSV file.
    :type output_file: str

    **Example**::

        >>> from detecto.utils import xml_to_csv

        >>> xml_to_csv('xml_labels/', 'labels.csv')
    """

    xml_list = []
    # Loop through every XML file
    for xml_file in glob(xml_folder + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Each object represents each actual image label
        for member in root.findall('object'):
            box = member.find('bndbox')
            label = member.find('name').text

            # Add image file name, image size, label, and box coordinates to CSV file
            row = (filename, width, height, label, int(box[0].text),
                   int(box[1].text), int(box[2].text), int(box[3].text))
            xml_list.append(row)

    # Save as a CSV file
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_names)
    xml_df.to_csv(output_file, index=None)


# Checks whether a variable is a list or tuple only
def _is_iterable(variable):
    return isinstance(variable, list) or isinstance(variable, tuple)
