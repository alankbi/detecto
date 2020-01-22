Quickstart
==========

Installation
------------

Detecto can be installed with pip::

    pip3 install detecto

Technical Requirements
----------------------

By default, Detecto will run all heavy-duty code on the GPU if it's available
and on the CPU otherwise. However, training and inference can take a long
time without a GPU. Thus, if your computer doesn't have a GPU you can use,
consider using a service such as `Google Colab
<https://colab.research.google.com/>`_, which comes with a free GPU.

Check out the `demo on Colab
<https://colab.research.google.com/drive/1ISaTV5F-7b4i2QqtjTa7ToDPQ2k8qEe0>`_
to learn more about both Detecto and Colab!

Data Format
-----------

Before starting, you should have a labeled dataset of images. If you don't
have one, you can download a dataset of Chihuahuas and Golden Retrievers
:download:`here <../_static/dog_dataset.zip>`. This dataset is a modified
subset of the `Stanford Dogs Dataset
<http://vision.stanford.edu/aditya86/ImageNetDogs/>`_.

The label data should be in individual XML files that each correspond to
one image. To label your images and create these XML files, see `LabelImg
<https://github.com/tzutalin/labelImg>`_, a free and open source tool that
makes it easy to label your data and produces XML files in the PASCAL VOC
format that Detecto uses. In the future, more formats for label data will
be supported.

Your data may look like the following::

    your_folder/
    |   image1.jpg
    |   image1.xml
    |   image2.jpg
    |   image2.xml
    |   ...

Or like the following::

    your_images_folder/
    |   image1.jpg
    |   image2.jpg
    |   ...

    your_xml_folder/
    |   image1.xml
    |   image2.xml
    |   ...

If you'd like to split your data into a training set and a validation set,
the images can be in the same folder, but the XML files should be in
separate folders::

    your_images_folder/
    |   image1.jpg
    |   image2.jpg
    |   ...

    train_labels/
    |   image1.xml
    |   image3.xml
    |   ...

    test_labels/
    |   image2.xml
    |   image4.xml
    |   ...

Code
----

First, check that you can read in and plot an image::

    import matplotlib.pyplot as plt
    from detecto.utils import read_image

    image = read_image('path_to_image.jpg')
    plt.imshow(image)
    plt.show()

Next, convert your XML label files into a CSV file. This allows us to create
a Dataset of our images that we can index over, as you'll see later::

    from detecto.core import Dataset
    from detecto.utils import xml_to_csv

    xml_to_csv('your_xml_folder', 'your_output_file.csv')
    dataset = Dataset('your_output_file.csv', 'your_images/')

Alternatively, apply some `custom transforms
<https://pytorch.org/docs/stable/torchvision/transforms.html>`_ on your dataset
for purposes such as data augmentation. If you choose to supply your own
transforms, note that you must convert the images to torch.Tensors and normalize
them at the very end. In the below example, we define a torchvision Compose object
that tells our dataset to convert images to PIL images, apply resize, flip, and
saturation augmentations, and then finally convert back to normalized tensors::

    from torchvision import transforms
    from detecto.utils import normalize_transform

    custom_transforms = transforms.Compose([
        transforms.ToPILImage(),
        # Note: all images with a size smaller than 800 will be scaled up in size
        transforms.Resize(800),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(saturation=0.2),
        transforms.ToTensor(),  # required
        normalize_transform(),  # required
    ])
    dataset = Dataset('your_output_file.csv', 'your_images/', transform=custom_transforms)

Let's check to make sure we have a working dataset; when we index it, we should
receive a tuple of the image and a dict containing label and box data. As the
dataset normalizes our images, the :func:`detecto.visualize.show_labeled_image`
automatically applies a reverse-normalization to restore it as close to the
original as possible::

    from detecto.visualize import show_labeled_image

    image, targets = dataset[0]
    show_labeled_image(image, targets['boxes'])

Now, let's train a model on our dataset. First, specify what classes you
want to predict when initializing the Model. After that, we'll need
to create a DataLoader over our dataset; because image datasets are typically
very large, the model can only train on it in smaller batches. The DataLoader
helps define how we batch and feed our images into the model for training::

    from detecto.core import DataLoader, Model

    # Specify all unique labels you're trying to predict
    your_labels = ['label1', 'label2', '...']
    model = Model(your_labels)

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model.fit(loader, verbose=True)

Optionally, supply a validation dataset to track accuracy throughout training
and tweak some of the training options::

    val_dataset = Dataset('your_val_labels.csv', 'your_val_images/')
    val_loader = DataLoader(val_dataset)
    losses = model.fit(loader, val_loader, epochs=15, learning_rate=0.01,
                       gamma=0.2, lr_step_size=5, verbose=True)

    plt.plot(losses)
    plt.show()

The model is finally ready for inference! You can pass in a single image or a
list of images to the model's predict methods, and you can choose to receive
all predictions or just the top ones per label::

    image = read_image('path_to_image.jpg')
    predictions = model.predict(image)

    images = []
    for i in range(4):
        image, _ = val_dataset[i]
        images.append(image)

    top_predictions = model.predict_top(images)

    print(predictions)
    print(top_predictions)

Lastly, we can plot a grid of predictions across several images or generate a
video with real-time object detection::

    from detecto.visualize import plot_prediction_grid, detect_video

    plot_prediction_grid(model, images, dim=(2, 2), figsize=(8, 8))
    detect_video(model, 'your_input_video.mp4', 'your_output_file.avi')

For next steps, see the :ref:`Further Usage <further-usage>` tutorial.
