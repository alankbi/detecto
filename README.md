# Gate Detection with PyTorch

This markdown file provides instruction and documentation for the [Gate Detection notebook](https://colab.research.google.com/drive/1gSTGj93GdHxyOBii1vLLfmRgraL7M8vo) for Duke Robotics. Before running the notebook, read the intro and do a CTRL+F search for "ATTENTION" to see which cells require your attention before the notebook can run smoothly.  

## Background

The goal is to create an autonomous robot that is able to navigate various underwater obstacles. One of these tasks is moving through a gate, which has a black horizontal bar on top and two orange vertical bars on the side to denote where to swim through. The gate also has an orange tick hanging from the top that splits the gate roughly in half, and you can earn extra points by moving through the smaller half. In this project, we use object detection with machine learning to detect the boundaries of the gate and the tick. 

## Technologies

The code is written in Python and hosted in a [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) notebook. We use [PyTorch](https://pytorch.org/) and PyTorch's [torchvision](https://pytorch.org/docs/stable/torchvision/index.html) package to do most of the machine learning work. [scikit-image](https://scikit-image.org/) is for reading in images, and [matplotlib](https://matplotlib.org/) is for visualizing all the images and predictions we generate. Other libraries specific to certain code cells are imported when needed.

## Notebook Overview

First, we start off importing the necessary packages and mounting our Drive, meaning we give the notebook access to the files on our Google Drive account. From there, we navigate to the folder storing all our image and label data (this location depends on the structure of your Drive). After reading in, plotting, and applying a few preliminary transforms to an image from our dataset (to confirm things are working), we're ready to start cleaning and processing our data to later feed into our model. 

First, the label data we have are in individual XML files, so we run a script to convert and save them all into a single CSV file than can be read in as a [pandas](https://pandas.pydata.org/) DataFrame object. We then create a class called GateDataset, which we pass our paths to the image/label data and can then use to easily index our data (i.e. Python indexing: running dataset[i] will return a tuple containing the i'th image and label data). 

After plotting some images and boxes from our dataset, we import a pre-trained Faster R-CNN model. As is, the model can't prediction our custom labels (start_tick and start_gate), so we'll need to do further training, just on the head of the model. We replace the model's box predictor with a FastRCNNPredictor and train it on our data on Colab's GPU, which takes anywhere from two minutes to an hour, depending on how many epochs you run it for. 

Finally, we can test our model by plotting its predictions on some of the test data or running it on a video, which generates a new video with live object detection in every frame. If our model performs well, we can save it to a .pth file and load it in for future use. 

## Documentation

###  Classes:

#### `class GateDataset(csv_file, root_dir, transform=None)`

A class that takes in the paths to your image/label data and allows you to traverse your dataset using Python indexing. An optional [transform.Compose](https://pytorch.org/docs/stable/torchvision/transforms.html) object can be applied to each item when returned. Implementation detail: images are read in only when specifically indexed to prevent having hundreds of megabytes of images loaded into memory at once. 

**Parameters:**

- `csv_file`: The path to your CSV file containing the label data. The file should have the following columns in order: `filename, width, height, class, xmin, ymin, xmax, ymax`. Each row represents one set of labeled class and box coordinates for an image. 

- `root_dir`: The path to the directory containing the images. 
- `transform`: Optional `transform.Compose` object containing the transforms to be applied to every data item. If none is given, no transforms will be applied. 

**Indexing:**

When indexed, the GateDataset object returns a tuple of size two: the first element is the image and the second element is a dictionary containing a 'boxes' and 'labels' key. The 'boxes' key contains a tensor of size [1, 4] representing the xmin, ymin, xmax, and ymax of the box, and labels is a tensor of size [1] containing the label's class. 

**Example Usage:**

```python
img_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(108),
    transforms.ToTensor(), 
])

dataset = GateDataset('labels.csv', 'images/', transform=img_transforms)
image, target = dataset[0]
print(image.shape, target['boxes'], target['labels'])
# Example output: (1080, 1720, 3), tensor([[1114,  466, 1153,  627]]), tensor([2])
```

### Functions

#### `def collate_data(batch)`

Utility function converting a list of the tuples returned from our custom GateDataset into a tuple of lists. Its primary usage is to tell a [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) how to properly batch data from a GateDataset into the format required to feed into the model; without passing this function into the DataLoader's `collate_fn` parameter, the train function will throw an error when you pass it the data loader. 

**Parameters:**

- `batch`: A list of the tuples returned by indexing a GateDataset.

**Return:**

A tuple (size 2) of lists, the first list containing the images and the second list containing the targets dictionaries returned by indexing a GateDataset. 

**Example Usage:**

```python
dataset = GateDataset('labels.csv', 'images/', transform=None)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_data)
```



#### `def detect_video(model, device, input_file, output_file)`

Takes in a source video and generates a new video with live object detection at every frame. 

**Parameters:**

- `model`: The model to use for predictions.
- `device`: The device on which to run the predictions. 
- `input_file`: The path to the input video file. 
- `output_file`: Where to write the output video. The output file must be in .avi format. 

**Example Usage:**

```python
... # Code to create model and device
detect_video(model, device, 'input_video.mp4', 'out/video1.avi')
```



####  `def filter_top_predictions(labels, boxes, scores)`

Filters out the top scoring predictions of each class from the given data. 

**Parameters:**

- `labels`: A list containing the string labels.
- `boxes`: A tensor of size [N, 4] containing the box coordinates.
- `scores`: A tensor of size N containing the score for each prediction. 

**Return:**

Returns a list of size K, where K is the number of uniquely predicted classes in labels. Each element in the list is a tuple containing the label, a tensor of size 4 containing the box coordinates, and the score for that prediction. 

**Example Usage:**

```python
... # Code to create model, image, and device
labels, boxes, scores, _ = get_clean_predictions(model, image, device)
top_preds = filter_top_predictions(labels, boxes, scores)
top_label, top_box, top_score = top_preds[0]
```



#### `def get_clean_predictions(model, images, device)`

Get predictions for a single image. 

**Parameters:**

- `model`: The model to use for predictions.
- `images`: An image or list of images on which to run object detection.
- `device`: The device to run the code on. 

**Return:**

If images is a list of images, returns a list of size N (the number of images), with each element being a list of size 4. Otherwise, if images is a single image, returns a single list of size 4. Each list of size 4 contains 1. a list of size K (number of predicted objects within the current image) containing the string labels for each predicted object, 2. a tensor of size [K, 4] containing the box coordinates, 3. a tensor of size K containing the prediction scores, and 4. the image itself. 

**Example Usage:**

```python
from skimage import io

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = ... # See 'Example Code' section for info on creating the model
image = io.imread('image.png')

labels, boxes, scores, image = get_clean_predictions(model, image, device)
```



#### `def int_to_label(val)`

Maps integers to labels after receiving predictions from a model. 

**Parameters:**

- `val`: The integer value of a class.

**Return:**

Returns the string label associated with the given integer.

**Example Usage:**

```python
print(int_to_label(1)) # Prints 'start_gate'
```



#### `def label_to_int(label)`

Maps string labels to integers in preparation for feeding into a model. 

**Parameters:**

- `label`: The path to the directory containing the XML files. 

**Return:**

Returns an integer ranging from 0 to the number of classes. 

**Example Usage:**

```python
print(label_to_int('start_gate')) # Prints 1
```



#### `def load_model(path, device, num_classes)`

Load model from saved weights. 

**Parameters:**

- `path`: Path to the .pth file containing the saved weights file. 
- `device`: The device to send the model to. 
- `num_classes`: The number of classes the model was trained to predict (including the default background). 

**Return:**

Returns the loaded model. 

**Example Usage:**

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = load_model('model_weights.pth', device, 3)
```



#### `def plot_prediction_grid(model, dataset, device, start=0, step=2)`

Creates a 3x3 grid of plots containing the model's predictions over certain images in a dataset. 

**Parameters:**

- `model`: The model to use for predictions.
- `dataset`: A GateDataset object containing the gate images. 
- `device`: The device on which to run the predictions. 
- `start`: Starting index of the gate dataset to predict on. Defaults to 0.
- `step`: How far to jump ahead through the dataset for each successive plot in the grid. Defaults to 2.

**Example Usage:**

```python
... # Code to create model, dataset, and device
plot_prediction_grid(model, dataset, device, start=20)
```



#### `def reverse_normalize(image)`

Undos the default normalize transform on an image to return it to its original self (usually so that it can be plotted). The default normalization for PyTorch's pre-trained models is the following:

`transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255], std=[1 / 0.229, 1 / 0.224, 1 / 0.255])`

**Parameters:**

- `image`: The normalized image.

**Return:**

Returns the image without normalization. 

**Example Usage:**

```python
from skimage import io

image = io.imread('image.png')

img_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transformed_image = img_transforms(image)
reverse_normalized_image = reverse_normalize(transformed_image)
```



#### `def save_model(model, path)`

Saves the model weights to a .pth file. 

**Parameters:**

- `model`: The model to save.
- `path`: The file path to save the model weights to. 

**Example Usage:**

```python
model = ... # See 'Example Code' section for info on creating the model
save_model(model, 'model_weights.pth')
```



#### `def show_labeled_image(image, boxes)`

Plots an image along with boxes around detected objects. 

**Parameters:**

- `image`: The image to plot as either a valid tensor or PILImage. 
- `boxes`: A list/tensor of size [4] or [N, 4] containing the xmin, ymin, xmax, and ymax coordinates of the boxes. N is the number of boxes to plot.

**Example Usage:**

```python
from skimage import io

image = io.imread('image.png')
boxes = torch.tensor([[100, 200, 550, 750], [320, 80, 760, 480]])
show_labeled_image(image, boxes)
```



#### `def train_model(model, data_loader, device, epochs=10, learning_rate=0.005, lr_step_size=3)`

Trains a machine learning model.

**Parameters:**

- `model`: The model to train.
- `data_loader`: A [DataLoader](https://pytorch.org/docs/stable/data.html) object containing the training data.
- `device`: The device on which to run the training.
- `epochs`: The number of runs over the entire dataset on which to train the model. Defaults to 10.
- `learning_rate`: Initial learning rate with which to apply updates to the model weights. Defaults to 0.005. 
- `lr_step_size`: Number of epochs to run before decreasing the learning rate by 10x. Each successive lr_step_size epochs decreases the learning rate by another 10x. Defaults to 3. 

**Return:**

Returns a list of the loss values at each step during training. 

**Example Usage:**

```python
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = ... # See 'Example Code' section for info on creating the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

dataset = GateDataset(...) # See GateDataset docs
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_data) # See notebook source code for info on collate_data

losses = train_model(model, loader, device, epochs=5)
```



#### `def xml_to_csv(path, output_path)`

Converts a directory of XML label files into a CSV file. The CSV file has the following columns in order: `filename, width, height, class, xmin, ymin, xmax, ymax`. Each row represents one set of labeled class and box coordinates for an image. 

**Parameters:**

- `path`: The path to the directory containing the XML files. 

- `output_path`: The output location for the CSV file. 

**Example Usage:**

```python
xml_to_csv('xml_labels/', 'labels.csv')
```



## Example Code

Below is some example code showing how to make use of the classes and functions created in the notebook. 

```python
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt

# Convert your XML files into a single CSV file
xml_to_csv('xml_labels/', 'labels.csv')

# Define the transforms you want to apply to your images
transform_img = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(108),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(saturation=0.5),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the dataset and data loader 
dataset = GateDataset('labels.csv', 'images/', transform=transform_img)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_data)

# Load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Number of classes we want the fine-tuned model to predict
num_classes = 3
# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Send model to the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# Train the model
losses = train_model(model, loader, device, epochs=1)

# Plot loss over time
plt.plot(losses)
plt.show()

# Plot some of the predictions made by the model
plot_prediction_grid(model, dataset, device)

# Generate a video with live object detection
detect_video(model, device, 'input_video.mp4', 'output_video.avi')
```

