#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[79]:


# Imports here
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets, transforms, models

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from collections import OrderedDict
import time
import random, os
import seaborn as sns

from PIL import Image

import json
import time


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[3]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[4]:


# TODO: Define your transforms for the training, validation, and testing sets
training_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

validataion_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])])

testing_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])]) 

# TODO: Load the datasets with ImageFolder
image_datasets = [datasets.ImageFolder(train_dir, transform=training_transforms), # can use indexing to select train/val/test
                  datasets.ImageFolder(valid_dir, transform=validataion_transforms),
                  datasets.ImageFolder(test_dir, transform=testing_transforms)]

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
               torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True),
               torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)]


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[5]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[6]:


# TODO: Build and train your network
model = models.densenet121(pretrained=True)
model


# In[7]:


for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([ # use sequential to save time 
                          ('fc1', nn.Linear(1024, 500)),
                          ('dropout', nn.Dropout(p=0.6)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
model.classifier = classifier
model


# In[ ]:





# In[8]:


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

epochs = 3
print_every = 10
steps = 0

# change to cuda
model.to('cuda') # use cuda


start = time.time()
print('Starting training')

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(dataloaders[0]): # 0 = train
        steps += 1 
        
        inputs, labels = inputs.to('cuda'), labels.to('cuda') # use cuda
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            valloss = 0
            accuracy=0
        
            for ii, (inputs2,labels2) in enumerate(dataloaders[1]): # 1 = validation 
                    optimizer.zero_grad()

                    inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda') # use cuda
                    model.to('cuda:0') # use cuda
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        valloss = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

            valloss = valloss / len(dataloaders[1])
            accuracy = accuracy /len(dataloaders[1])

            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Training Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Loss {:.4f}".format(valloss),
                  "Accuracy: {:.4f}".format(accuracy),
                 )

            running_loss = 0
            
time_elapsed = time.time() - start
print("\nTime spent training: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[15]:


# TODO: Do validation on the test set
correct = 0
total = 0
model.to('cuda')
with torch.no_grad():
    for data in dataloaders[2]: # 2 = testing
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[19]:


# TODO: Save the checkpoint 
model.class_to_idx = image_datasets[0].class_to_idx

checkpoint = {'input_size': 1024,
              'output_size': 102,
              'arch': 'densenet121',
              'classifier' : classifier,
              'learning_rate': 0.001,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
             }

torch.save(checkpoint, 'checkpoint.pth')


# In[51]:


filename = 'checkpoint.pth'
filename


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[56]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    
   # Load model metadata
    # Loading weights for CPU model whoch were trained on GPU
    # https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    # Recreate the pretrained base model
    #model = models.vgg16(pretrained=True)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    # Replace the classifier part of the model
    model.classifier = checkpoint['classifier']
    
    # Rebuild saved state dict
    model.load_state_dict(checkpoint['state_dict'])

    # Load class_to_idx
    model.class_to_idx = checkpoint['class_to_idx']

    return model


# In[57]:


model_from_file = load_checkpoint(filename)
model_from_file


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[58]:


# Tests image display before transformations
image_path = 'flowers/test/10/image_07090.jpg'
img = Image.open(image_path)
img


# In[59]:


def process_image(image):
    

    
    
    # TODO: Process a PIL image for use in a PyTorch model
    # Find the shorter side and resize it to 256 keeping aspect ration
    # if the width > height
    if image.width > image.height:        
        # Constrain the height to be 256
        image.thumbnail((10000000, 256))
    else:
        # Constrain the width to be 256
        image.thumbnail((256, 10000000))
    
    # Center crop the image
    crop_size = 224
    left_margin = (image.width - crop_size) / 2
    bottom_margin = (image.height - crop_size) / 2
    right_margin = left_margin + crop_size
    top_margin = bottom_margin + crop_size  
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Convert values to range of 0 to 1 instead of 0-255
    image = np.array(image)
    image = image / 255
    
    # Standardize values
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image = (image - means) / stds
    
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose(2, 0, 1)
    
    return image


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[60]:


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    if title:
        plt.title(title)
    
    ax.imshow(image)
    
    return ax


# In[61]:


# Test image processing & show the transformed image
img_ready = process_image(img)
imshow(img_ready);


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[62]:



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Move model into evaluation mode and to CPU
    model.eval()
    model.cpu()
   
    # Open image
    image = Image.open(image_path)
    
    # Process image
    image = process_image(image) 
    
    # Change numpy array type to a PyTorch tensor
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    
    # Format tensor for input into model
    # (add batch of size 1 to image)
    # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
    image = image.unsqueeze(0)
    
    # Predict top K probabilities
    # Reverse the log conversion
    probs = torch.exp(model.forward(image))
    top_probs, top_labs = probs.topk(topk)
    #print(top_probs)
    #print(top_labs)
    
    # Convert from Tesors to Numpy arrays
    top_probs = top_probs.detach().numpy().tolist()[0]
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    # Map tensor indexes to classes
    labs = []
    for label in top_labs.numpy()[0]:
        labs.append(idx_to_class[label])

    return top_probs, labs


# In[63]:


# Predict class of an image
probs, classes = predict(image_path, model_from_file, topk=5)
print(image_path)
print(probs)
print(classes)


# In[73]:


def convertCategoryToName(categories, mapper='cat_to_name.json'):
    
    # Load json file
    with open(mapper, 'r') as f:
        cat_to_name = json.load(f)
        #print(cat_to_name)
    
        names = []

        # Find flower names corresponding to predicted categories
        for category in categories:
            names.append(cat_to_name[str(category)])

    return names


# In[74]:


names = convertCategoryToName(classes)
print(probs)
print(classes)
print(names)


# In[75]:


# sanity check
img = ('flowers/test/1/image_06743.jpg') # should be primrose, class = 1
probs, classes = predict(img, model)
print(probs)
print(classes)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[76]:


def select_random_image(path):
    
    # Select random file
    random_folder = random.choice(os.listdir(path))
    random_file = random.choice(
    os.listdir(os.path.join(path, random_folder)))

    # Create image path
    image_path = os.path.join(path, random_folder, random_file)

    return image_path


# In[80]:


def display_preds(path, model, topk=5, flower_names=None):

    # Return random path to an image
    image_path = select_random_image(path)
    print(image_path)
    
    # Return folder number which equals to 
    # a class identifier
    folder_number = image_path.split('/')[2]
    #print(folder_number)
    
    # Read the flower name based on the folder
    # number (flower class id) and external dictionary mapping 
    title = flower_names[folder_number]
    #print(title)
    
    # Predict image class
    probs, classes = predict(image_path, model, topk)
    #print(probs)
    #print(classes)
    
    # Convert class id into its name
    names = convertCategoryToName(classes)
    print('n:', names)
    print('c:', classes)
    
    # Open an image
    image = Image.open(image_path)
    
    # Make the image compliant with PyTorch
    image = process_image(image)
    
    # Set up a plot
    plt.figure(figsize = (6, 10))
    ax = plt.subplot(2, 1, 1)

    # Plot the flower
    imshow(image, ax, title=title);

    # Visualize prediction result
    plt.subplot(2, 1, 2)
    sns.barplot(x=probs, y=names, color=sns.color_palette()[0]);
    plt.show()

    return


# In[81]:


# TODO: Display an image along with the top 5 classes
 #Perform prediction on a number of images
number_of_predictions = 5

for i in range(number_of_predictions):
    display_preds('flowers/test', model_from_file, flower_names=cat_to_name)


# In[ ]:





# In[ ]:




