.. Detecto documentation master file, created by
   sphinx-quickstart on Thu Dec 26 16:43:20 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Detecto's documentation!
===================================

`Detecto <https://github.com/alankbi/detecto>`_ is a Python package that
allows you to build fully-functioning computer vision and object detection
models with just 5 lines of code. Inference on still images and videos,
transfer learning on custom datasets, and serialization of models to files
are just a few of Detecto's features. Detecto is also built on top of PyTorch,
allowing an easy transfer of models between the two libraries.

The power of Detecto comes from its simplicity and ease of use. Creating and
running a pre-trained `Faster R-CNN ResNet-50 FPN
<https://pytorch.org/docs/stable/torchvision/models.html
#object-detection-instance-segmentation-and-person-keypoint-detection>`__ from
PyTorch's model zoo takes 4 lines of code::

   from detecto.core import Model
   from detecto.visualize import detect_video

   model = Model()
   detect_video(model, 'input.mp4', 'output.avi')

Below are several more examples of things you can do with Detecto:

**Transfer Learning on Custom Datasets**

Most of the times, you want a computer vision model that can detect custom
objects. With Detecto, you can train a model on a custom dataset with 5
lines of code::

   from detecto.core import Model, Dataset

   dataset = Dataset('custom_dataset/')

   model = Model(['dog', 'cat', 'rabbit'])
   model.fit(dataset)

   model.predict(...)


**Inference and Visualization**

When using a model for inference, Detecto returns predictions in an
easy-to-use format and provides several visualization tools::

   from detecto.core import Model
   from detecto import utils, visualize

   model = Model()

   image = utils.read_image('image.jpg')

   # Model's predict and predict_top methods

   labels, boxes, scores = model.predict(image)
   predictions = model.predict_top(image)

   print(labels, boxes, scores)
   print(predictions)

   # Visualize module's helper functions

   visualize.show_labeled_image(image, boxes, labels)

   images = [...]
   visualize.plot_prediction_grid(model, images)

   visualize.detect_video(model, 'input_video.mp4', 'output.avi')
   visualize.detect_live(model)

**Advanced Usage**

If you want more control over how you train your model, Detecto lets you
do just that::

   from detecto import core, utils
   from torchvision import transforms
   import matplotlib.pyplot as plt

   # Change data format

   utils.xml_to_csv('training_labels/', 'train_labels.csv')
   utils.xml_to_csv('validation_labels/', 'val_labels.csv')

   # Custom transforms

   custom_transforms = transforms.Compose([
       transforms.ToPILImage(),
       transforms.Resize(800),
       transforms.ColorJitter(saturation=0.3),
       transforms.ToTensor(),
       utils.normalize_transform(),
   ])

   dataset = core.Dataset('train_labels.csv', 'images/',
                          transform=custom_transforms)

   # Validation dataset

   val_dataset = core.Dataset('val_labels.csv', 'val_images')

   # Customize training options

   loader = core.DataLoader(dataset, batch_size=2, shuffle=True)

   model = core.Model(['car', 'truck', 'boat', 'plane'])
   losses = model.fit(loader, val_dataset, epochs=15,
                      learning_rate=0.001, verbose=True)

   # Visualize loss during training

   plt.plot(losses)
   plt.show()

   # Save model

   model.save('model_weights.pth')

   # Access underlying torchvision model for further control

   torch_model = model.get_internal_model()
   print(type(torch_model))

For a deeper dive into the package, see the guides below to get started, or
check out the `demo on Colab
<https://colab.research.google.com/drive/1ISaTV5F-7b4i2QqtjTa7ToDPQ2k8qEe0>`_:

.. toctree::
   :maxdepth: 2

   usage/quickstart
   usage/further-usage


API Documentation
=================

.. toctree::
   :titlesonly:
   :maxdepth: 2

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
