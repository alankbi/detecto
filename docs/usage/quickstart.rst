Quickstart
==========

Installation
------------

Detecto can be installed with pip::

    pip3 install detecto

Technical Requirements
----------------------

By default, Detecto will run all heavy-duty code on the GPU if it's available
and on the CPU otherwise. However, training and even inference can take a long
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
<http://vision.stanford.edu/aditya86/ImageNetDogs/>`_. If you have a video
you'd like to use as training data, you can use
:func:`detecto.utils.split_video` to split it into individual images that
you can then label.

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

Above is the recommended way to store your data. However, other formats work
as well, such as the following::

    your_images_folder/
    |   image1.jpg
    |   image2.jpg
    |   ...

    your_xml_folder/
    |   image1.xml
    |   image2.xml
    |   ...

If you'd like to split your data into a training set and a validation set,
you could have two separate folders like so::

    training_data/
    |   image1.jpg
    |   image1.xml
    |   image2.jpg
    |   image2.xml
    |   ...

    validation_data/
    |   image3.jpg
    |   image3.xml
    |   image4.jpg
    |   image4.xml
    |   ...

Or, you can have all the images in the same folder but the XML files in
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

Note that your images and XML files don't need to share the same file name.
However, doing so can make associations between files clearer.

Code
----

First, check that you can read in and plot an image::

    import matplotlib.pyplot as plt
    from detecto.utils import read_image

    image = read_image('path_to_image.jpg')
    plt.imshow(image)
    plt.show()

Next, create a Dataset object from your images and label data::

    from detecto.core import Dataset

    # If your images and labels are in the same folder
    dataset = Dataset('your_images_and_labels/')
    # If your images and labels are in separate folders
    dataset = Dataset('your_labels/', 'your_images/')

If you plan to make many runs over your training data, you may want
to generate a CSV file from your XML data. Then, whenever you create a
Dataset, you can pass it this CSV file instead of your folder of XML
files. This may make it a bit easier to work with your data in the future::

    from detecto.utils import xml_to_csv

    xml_to_csv('your_labels/', 'labels.csv')
    dataset = Dataset('labels.csv', 'your_images/')

In addition, you can apply many `custom transforms
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
    dataset = Dataset('your_training_data/', transform=custom_transforms)

Let's check to make sure we have a working dataset; when we index it, we should
receive a tuple of the image and a dict containing label and box data. As the
dataset normalizes our images, the :func:`detecto.visualize.show_labeled_image`
automatically applies a reverse-normalization to restore it as close to the
original as possible::

    from detecto.visualize import show_labeled_image

    image, targets = dataset[0]
    show_labeled_image(image, targets['boxes'], targets['labels'])

Now, let's train a model on our dataset. First, specify what classes you
want to predict when initializing the Model. After that, you can optionally
create a DataLoader over your Dataset; because image datasets are typically
very large, the model can only train on it in smaller batches. The DataLoader
helps define how we batch and feed our images into the model for training. If
you decide not to provide your own DataLoader, the model with automatically
wrap your dataset in a default DataLoader when training::

    from detecto.core import DataLoader, Model

    # Specify all unique labels you're trying to predict
    your_labels = ['label1', 'label2', '...']
    model = Model(your_labels)

    model.fit(dataset, verbose=True)

    # Alternatively, provide your own DataLoader to the fit method
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model.fit(loader, verbose=True)

You can also supply a validation dataset to track accuracy throughout training
as well as tweak some of the training parameters::

    val_dataset = Dataset('validation_dataset/')
    losses = model.fit(dataset, val_dataset, epochs=15, learning_rate=0.01,
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

Lastly, we can plot a grid of predictions across several images, generate a
video with real-time object detection, or run predictions on a live webcam::

    from detecto.visualize import plot_prediction_grid, detect_video, detect_live

    plot_prediction_grid(model, images, dim=(2, 2), figsize=(8, 8))
    detect_video(model, 'your_input_video.mp4', 'your_output_file.avi')
    detect_live(model, score_filter=0.7)  # Note: may not work on VMs

For next steps, see the :ref:`Further Usage <further-usage>` tutorial.
