.. _further-usage:

Further Usage
=============

Saving and Loading
------------------

After training a model you're happy with, saving and loading it from a file
is easy. To save a model to a file (the recommended file extension is .pth),
use the :meth:`save <detecto.core.Model.save>` method::

    from detecto.core import Model

    labels = ['label1', 'label2', '...']
    model = Model(labels)
    # ... training and other steps ...

    model.save('your_save_file.pth')

To load a model from a file, use the static :meth:`load <detecto.core.Model.load>`
method::

    model = Model.load('your_save_file.pth', labels)

Be sure that the list of labels you provide is in the same order as when you
first initialized and saved the model.

Beyond Detecto
--------------

Detecto abstracts away a lot of the details of machine learning, and at a
certain point, you may decide you want more control. Since Detecto is
built on top of PyTorch and torchvision, transitioning to these feature-rich
libraries is easy. Simply use the :meth:`get_internal_model
<detecto.core.Model.get_internal_model>` method to access the underlying
torchvision model that Model uses::

    torch_model = model.get_internal_model()
    print(type(torch_model))

The internal model is a `Faster R-CNN architecture
<https://pytorch.org/docs/stable/torchvision/models.html
#object-detection-instance-segmentation-and-person-keypoint-detection>`_
with a FastRCNNPredictor box predictor. With the torchvision model itself,
you can now fine-tune the model accuracy, modify the model architecture,
and do many more things using the various PyTorch and torchvision modules.

For example, the following code limits fine-tuning during training to only
the last few layers of the model::

    for name, p in torch_model.named_parameters():
        print(name, p.requires_grad)

        if 'roi_heads' not in name and 'rpn' not in name:
            p.requires_grad = False

    # Can then proceed to train your Detecto model as usual
    model.fit(...)
