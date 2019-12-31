Quickstart
==========

Installation
------------

Detecto can be installed with pip::

    pip install detecto

Technical Requirements
----------------------

By default, Detecto will run all heavy-duty code on the GPU if it's available
and on the CPU otherwise. However, training and inference can take a long
time without a GPU. Thus, if your computer doesn't have GPU you can use,
consider using a service such as `Google Colab
<https://colab.research.google.com/>`_, which comes with a free GPU.

Data Format
-----------

Before starting, you should have a labeled dataset of images. The label data
should be in individual XML files that each correspond to one image. To
label your images and create these XML files, see `LabelImg
<https://github.com/tzutalin/labelImg>`_ a free and open source tool that
makes it easy to label your data and produces XML files in exactly the right
format for Detecto. In the future, more formats for label data will be
supported.

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

Here is some temporary code::

   from detecto.core import Model, Dataset, DataLoader
   from detecto.visualize import detect_video

   dataset = Dataset('labels.csv', 'images/')
   loader = DataLoader(dataset, batch_size=2, shuffle=True)

   model = Model(['dog', 'cat', 'rabbit'])
   model.fit(loader)

   detect_video(model, 'input_video.mp4', 'output_video.avi')

Here is some temporary text.
