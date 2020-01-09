import os
import pandas as pd
import random
import torch
import torchvision

from detecto.config import default_device
from detecto.utils import default_transforms, filter_top_predictions, _is_iterable
from skimage import io
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, **kwargs):
        """Accepts a :class:`detecto.core.Dataset` object and creates
        an iterable over the data, which can then be fed into a
        :class:`detecto.core.Model` for training and validation.
        Extends PyTorch's `DataLoader
        <https://pytorch.org/docs/stable/data.html>`_ class with a custom
        ``collate_fn`` function.

        :param dataset: The dataset for iteration over.
        :type dataset: detecto.core.Dataset
        :param kwargs: (Optional) Additional arguments to customize the
            DataLoader, such as ``batch_size`` or ``shuffle``. See `docs
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
            for more details.
        :type kwargs: Any

        **Example**::

            >>> from detecto.core import Dataset, DataLoader

            >>> dataset = Dataset('labels.csv', 'images/')
            >>> loader = DataLoader(dataset, batch_size=2, shuffle=True)
            >>> for images, targets in loader:
            >>>     print(images[0].shape)
            >>>     print(targets[0])
            torch.Size([3, 1080, 1720])
            {'boxes': tensor([[884, 387, 937, 784]]), 'labels': 'person'}
            torch.Size([3, 1080, 1720])
            {'boxes': tensor([[   1,  410, 1657, 1079]]), 'labels': 'car'}
            ...
        """

        super().__init__(dataset, collate_fn=DataLoader.collate_data, **kwargs)

    # Converts a list of tuples into a tuple of lists so that
    # it can properly be fed to the model for training
    @staticmethod
    def collate_data(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, image_folder, transform=None):
        """Takes in a CSV file containing label data and the path to the
        corresponding folder of images and creates an indexable dataset
        over all of the data. Applies optional transforms over the data.
        Extends PyTorch's `Dataset
        <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_.

        :param csv_file: Path to the CSV file containing the label data.
            The file should have the following columns in order:
            ``filename``, ``width``, ``height``, ``class``, ``xmin``,
            ``ymin``, ``xmax``, and ``ymax``. See
            :func:`detecto.utils.xml_to_csv` to generate CSV files in this
            format from XML label files.
        :type csv_file: str
        :param image_folder: The path to the folder containing images. Each
            row of the CSV file contains a ``filename`` which should
            correspond to an image in this folder.
        :type image_folder: str
        :param transform: (Optional) A torchvision `transforms.Compose
            <https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Compose>`__
            object containing transformations to apply on all elements in
            the dataset. See `PyTorch docs
            <https://pytorch.org/docs/stable/torchvision/transforms.html>`_
            for a list of possible transforms. When using transforms.Resize
            and transforms.RandomHorizontalFlip, all box coordinates are
            automatically adjusted to match the modified image. If None,
            defaults to the transforms returned by
            :func:`detecto.utils.default_transforms`.
        :type transform: torchvision.transforms.Compose or None

        **Indexing**:

        A Dataset object can be indexed like any other Python iterable.
        Doing so returns a tuple of length 2. The first element is the
        image and the second element is a dict containing a 'boxes' and
        'labels' key. ``dict['boxes']`` is a torch.Tensor of size
        ``(1, 4)`` containing ``xmin``, ``ymin``, ``xmax``, and ``ymax``
        of the box and ``dict['labels']`` is the string label of the
        detected object.

        **Example**::

            >>> from detecto.core import Dataset

            >>> dataset = Dataset('labels.csv', 'images/')
            >>> print(len(dataset))
            >>> image, target = dataset[0]
            >>> print(image.shape)
            >>> print(target)
            4
            torch.Size([3, 720, 1280])
            {'boxes': tensor([[564, 43, 736, 349]]), 'labels': 'balloon'}
        """

        # CSV file contains: filename, width, height, class, xmin, ymin, xmax, ymax
        self._csv = pd.read_csv(csv_file)

        self._root_dir = image_folder

        if transform is None:
            self.transform = default_transforms()
        else:
            self.transform = transform

    # Returns the length of this dataset
    def __len__(self):
        return len(self._csv)

    # Is what allows you to index the dataset, e.g. dataset[0]
    # dataset[index] returns a tuple containing the image and the targets dict
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read in the image from the file name in the 0th column
        img_name = os.path.join(self._root_dir, self._csv.iloc[idx, 0])
        image = io.imread(img_name)

        # Read in xmin, ymin, xmax, and ymax
        box = self._csv.iloc[idx, 4:]
        box = torch.tensor(box).view(1, 4)

        # Read in the label
        label = self._csv.iloc[idx, 3]

        targets = {'boxes': box, 'labels': label}

        # Perform transformations
        if self.transform:
            width = self._csv.loc[idx, 'width']
            height = self._csv.loc[idx, 'height']

            # Apply the transforms manually to be able to deal with
            # transforms like Resize or RandomHorizontalFlip
            updated_transforms = []
            scale_factor = 1.0
            random_flip = 0.0
            for t in self.transform.transforms:
                # Add each transformation to our list
                updated_transforms.append(t)

                # If a resize transformation exists, scale down the coordinates
                # of the box by the same amount as the resize
                if isinstance(t, transforms.Resize):
                    original_size = min(height, width)
                    scale_factor = original_size / t.size

                # If a horizontal flip transformation exists, get its probability
                # so we can apply it manually to both the image and the boxes.
                elif isinstance(t, transforms.RandomHorizontalFlip):
                    random_flip = t.p

            # Apply each transformation manually
            for t in updated_transforms:
                # Handle the horizontal flip case, where we need to apply
                # the transformation to both the image and the box labels
                if isinstance(t, transforms.RandomHorizontalFlip):
                    if random.random() < random_flip:
                        image = transforms.RandomHorizontalFlip(1)(image)
                        # Flip box's x-coordinates
                        box[0, 0] = width - box[0, 0]
                        box[0, 2] = width - box[0, 2]
                        box[0, 0], box[0, 2] = box[0, (2, 0)]
                else:
                    image = t(image)

            # Scale down box if necessary
            targets['boxes'] = (box / scale_factor).long()

        return image, targets


class Model:

    def __init__(self, classes, device=None):
        """Initializes a machine learning model for object detection.
        Models are built on top of PyTorch's `pre-trained models
        <https://pytorch.org/docs/stable/torchvision/models.html>`_,
        specifically the Faster R-CNN ResNet-50 FPN, but allow for
        fine-tuning to predict on custom classes/labels.

        :param classes: A list of classes/labels for the model to predict.
        :type classes: list
        :param device: (Optional) The device on which to run the model,
            such as the CPU or GPU. See `here
            <https://pytorch.org/docs/stable/tensor_attributes.html#torch-device>`_
            for details on specifying the device. Defaults to the GPU if
            available and the CPU if not.
        :type device: torch.device or None

        **Example**::

            >>> from detecto.core import Model

            >>> model = Model(['dog', 'cat', 'bunny'])
        """

        self._device = device if device else default_device

        # Load a model pre-trained on COCO
        self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # Get the number of input features for the classifier
        in_features = self._model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one (note: +1 because of the __background__ class)
        self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes) + 1)

        self._model.to(self._device)

        # Mappings to convert from string labels to ints and vice versa
        self._classes = ['__background__'] + classes
        self._int_mapping = {label: index for index, label in enumerate(self._classes)}

    # Returns the raw predictions from feeding an image or list of images into the model
    def _get_raw_predictions(self, images):
        self._model.eval()

        with torch.no_grad():
            # Convert image into a list of length 1 if not already a list
            if not _is_iterable(images):
                images = [images]

            # Convert to tensor and normalize if not already
            if not isinstance(images[0], torch.Tensor):
                defaults = default_transforms()
                images = [defaults(img) for img in images]

            # Send images to the specified device
            images = [img.to(self._device) for img in images]

            preds = self._model(images)
            # Send predictions to CPU if not already
            preds = [{k: v.to(torch.device('cpu')) for k, v in p.items()} for p in preds]
            return preds

    def predict(self, images):
        """Takes in an image or list of images and returns predictions
        for object locations.

        :param images: An image or list of images to predict on. If the
            images have not already been transformed into torch.Tensor
            objects, the default transformations contained in
            :func:`detecto.utils.default_transforms` will be applied.
        :type images: list or numpy.ndarray or torch.Tensor
        :return: If given a single image, returns a tuple of size
            three. The first element is a list of string labels of size N,
            the number of detected objects. The second element is a
            torch.Tensor of size (N, 4), giving the ``xmin``, ``ymin``,
            ``xmax``, and ``ymax`` coordinates of the boxes around each
            object. The third element is a torch.Tensor of size N containing
            the scores of each predicted object (ranges from 0.0 to 1.0). If
            given a list of images, returns a list of the tuples described
            above, each tuple corresponding to a single image.
        :rtype: tuple or list of tuple

        **Example**::

            >>> from detecto.core import Model
            >>> from detecto.utils import read_image

            >>> model = Model.load('model_weights.pth', ['horse', 'zebra'])
            >>> image = read_image('image.jpg')
            >>> labels, boxes, scores = model.predict(image)
            >>> print(labels[0])
            >>> print(boxes[0])
            >>> print(scores[0])
            horse
            tensor([   0.0000,  428.0744, 1617.1860, 1076.3607])
            tensor(0.9397)
        """

        # Convert all to lists but keep track if a single image was given
        is_single_image = not _is_iterable(images)
        images = [images] if is_single_image else images
        preds = self._get_raw_predictions(images)

        results = []
        for pred in preds:
            # Convert predicted ints into their corresponding string labels
            result = ([self._classes[val] for val in pred['labels']], pred['boxes'], pred['scores'])
            results.append(result)

        return results[0] if is_single_image else results

    def predict_top(self, images):
        """Takes in an image or list of images and returns the top
        scoring predictions for each detected label in each image.
        Equivalent to running :meth:`detecto.core.Model.predict` and
        then :func:`detecto.utils.filter_top_predictions` together.

        :param images: An image or list of images to predict on. If the
            images have not already been transformed into torch.Tensor
            objects, the default transformations contained in
            :func:`detecto.utils.default_transforms` will be applied.
        :type images: list or numpy.ndarray or torch.Tensor
        :return: If given a single image, returns a tuple of size
            three. The first element is a list of string labels of size K,
            the number of uniquely detected objects. The second element is
            a torch.Tensor of size (K, 4), giving the ``xmin``, ``ymin``,
            ``xmax``, and ``ymax`` coordinates of the top-scoring boxes
            around each unique object. The third element is a torch.Tensor
            of size K containing the scores of each uniquely predicted object
            (ranges from 0.0 to 1.0). If given a list of images, returns a
            list of the tuples described above, each tuple corresponding to
            a single image.
        :rtype: tuple or list of tuple


        **Example**::

            >>> from detecto.core import Model
            >>> from detecto.utils import read_image

            >>> model = Model.load('model_weights.pth', ['label1', 'label2'])
            >>> image = read_image('image.jpg')
            >>> top_preds = model.predict_top(image)
            >>> top_preds
            (['label2', 'label1'], tensor([[   0.0000,  428.0744, 1617.1860, 1076.3607],
            [ 875.3470,  412.1762,  949.5915,  793.3424]]), tensor([0.9397, 0.8686]))
        """

        predictions = self.predict(images)

        # If tuple but not list, then images is a single image
        if not isinstance(predictions, list):
            return filter_top_predictions(*predictions)

        results = []
        for pred in predictions:
            results.append(filter_top_predictions(*pred))

        return results

    def fit(self, data_loader, val_loader=None, epochs=10, learning_rate=0.005, momentum=0.9,
            weight_decay=0.0005, gamma=0.1, lr_step_size=3, verbose=False):
        """Train the model on the given :class:`detecto.core.DataLoader`.
        If given a DataLoader containing a validation dataset, returns a
        list of loss scores at each epoch.

        :param data_loader: A DataLoader containing the dataset to train on.
        :type data_loader: detecto.core.DataLoader
        :param val_loader: (Optional) A DataLoader containing the dataset
            to validate on. Defaults to None, in which case no validation
            occurs.
        :type val_loader: detecto.core.DataLoader
        :param epochs: (Optional) The number of runs over the data in
            ``data_loader`` to train for. Defaults to 10.
        :type epochs: int
        :param learning_rate: (Optional) How fast to update the model
            weights at each step of training. Defaults to 0.005.
        :type learning_rate: float
        :param momentum: (Optional) The momentum used to reduce the
            fluctuations of gradients at each step. Defaults to 0.9.
        :type momentum: float
        :param weight_decay: (Optional) The amount of L2 regularization
            to apply on model parameters. Defaults to 0.0005.
        :type weight_decay: float
        :param gamma: (Optional) The decay factor that ``learning_rate``
            is multiplied by every ``lr_step_size`` epochs. Defaults to 0.1.
        :type gamma: float
        :param lr_step_size: (Optional) The number of epochs between each
            decay of ``learning_rate`` by ``gamma``. Defaults to 3.
        :type lr_step_size: int
        :param verbose: (Optional) Whether to print the current epoch and
             loss (if given a validation dataset) at each step. Defaults
             to False.
        :type verbose: bool
        :return: If ``val_loader`` is not None and epochs is greater than 0,
            returns a list of the validation losses at each epoch. Otherwise,
            returns nothing.
        :rtype: list or None

        **Example**::

            >>> from detecto.core import Model, Dataset, DataLoader

            >>> dataset = Dataset('train_labels.csv', 'train_images/')
            >>> loader = DataLoader(dataset, batch_size=2, shuffle=True)
            >>> val_dataset = Dataset('val_labels.csv', 'val_images/')
            >>> val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            >>> model = Model(['rose', 'tulip'])
            >>> losses = model.fit(loader, val_loader, epochs=5)
            >>> losses
            [0.11191498369799327, 0.09899920264606253, 0.08454859235434461,
                0.06825731012780788, 0.06236840748117637]
        """

        losses = []
        # Get parameters that have grad turned on (i.e. parameters that should be trained)
        parameters = [p for p in self._model.parameters() if p.requires_grad]
        # Create an optimizer that uses SGD (stochastic gradient descent) to train the parameters
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        # Create a learning rate scheduler that decreases learning rate by gamma every lr_step_size epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

        # Train on the entire dataset for the specified number of times (epochs)
        for epoch in range(epochs):
            if verbose:
                print('Epoch {} of {}'.format(epoch + 1, epochs))

            # Training step
            self._model.train()
            for images, targets in data_loader:
                self._convert_to_int_labels(targets)
                images, targets = self._to_device(images, targets)

                # Calculate the model's loss (i.e. how well it does on the current
                # image and target, with a lower loss being better)
                loss_dict = self._model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())

                # Zero any old/existing gradients on the model's parameters
                optimizer.zero_grad()
                # Compute gradients for each parameter based on the current loss calculation
                total_loss.backward()
                # Update model parameters from gradients: param -= learning_rate * param.grad
                optimizer.step()

            # Validation step
            if val_loader is not None:
                avg_loss = 0
                with torch.no_grad():
                    for images, targets in data_loader:
                        self._convert_to_int_labels(targets)
                        images, targets = self._to_device(images, targets)
                        loss_dict = self._model(images, targets)
                        total_loss = sum(loss for loss in loss_dict.values())
                        avg_loss += total_loss.item()

                avg_loss /= len(val_loader.dataset)
                losses.append(avg_loss)

                if verbose:
                    print('Loss: {}'.format(avg_loss))

            # Update the learning rate every few epochs
            lr_scheduler.step()

        if len(losses) > 0:
            return losses

    def get_internal_model(self):
        """Returns the internal torchvision model that this class contains
        to allow for more advanced fine-tuning and the full use of
        features presented in the PyTorch library.

        :return: The torchvision model, which is a Faster R-CNN ResNet-50
            FPN with a FastRCNNPredictor box predictor.
        :rtype: torchvision.models.detection.faster_rcnn.FasterRCNN

        **Example**::

            >>> from detecto.core import Model

            >>> model = Model.load('model_weights.pth', ['tick', 'gate'])
            >>> torch_model = model.get_internal_model()
            >>> type(torch_model)
            <class 'torchvision.models.detection.faster_rcnn.FasterRCNN'>
        """

        return self._model

    def save(self, file):
        """Saves the internal model weights to a file.

        :param file: The name of the file. Should have a .pth file extension.
        :type file: str

        **Example**::

            >>> from detecto.core import Model

            >>> model = Model(['tree', 'bush', 'leaf'])
            >>> model.save('model_weights.pth')
        """

        torch.save(self._model.state_dict(), file)

    @staticmethod
    def load(file, classes):
        """Loads a model from a .pth file containing the model weights.

        :param file: The path to the .pth file containing the saved model.
        :type file: str
        :param classes: The list of classes/labels this model was trained
            to predict. Must be in the same order as initially passed to
            :meth:`detecto.core.Model.__init__` for accurate results.
        :type classes: list
        :return: The model loaded from the file.
        :rtype: detecto.core.Model

        **Example**::

            >>> from detecto.core import Model

            >>> model = Model.load('model_weights.pth', ['ant', 'bee'])
        """

        model = Model(classes)
        model._model.load_state_dict(torch.load(file, map_location=model._device))
        return model

    # Converts all string labels in a list of target dicts to
    # their corresponding int mappings
    def _convert_to_int_labels(self, targets):
        for target in targets:
            # Convert string labels to integer mapping
            target['labels'] = torch.tensor(self._int_mapping[target['labels']]).view(1)

    # Sends all images and targets to the same device as the model
    def _to_device(self, images, targets):
        images = [image.to(self._device) for image in images]
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]
        return images, targets
