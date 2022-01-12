# Prediction-of-COVID-19-from-chest-X-ray-images-using-Deep-Transfer-Learning
In this project, I have tried to build a system that can detect covid-19 patients from their chest x-ray images, where, the intelligent system can classify whether the patient is affected by covid or not from a given x-ray image. It is an image classification problem which fits under the domain of computer vision and deep learning. In the field of deep learning, various neural networks have developed which are able to process complex images and can make classification with higher accuracy rates.

Globally covid19 has caused a catastrophic effect on human lives, and this has continued ever since its outbreak. Thousands of people have lost their lives and hospitals are getting out of ICUs. According to the experts, identifying the covid patients as early as we can, may play a vital role in bringing the pandemic under control. PCR test, which is a covid-19 diagnostic test, needs at least 1-3 days to get the results to its patients. But to control the outbreak, we need to come up with faster methods that can produce results rapidly. So in quest of a rapid method that can identify covid-19 patients quickly, directed us to our current project. We tried to build a system, powered by artificial intelligence, that takes x-ray images as an input and gives the verdict of whether the patient is infected with coronavirus or not. 

Certainly this task is not easy. Usually deep learning models require thousands and thousands of images to make a strong, rigid and sustainable model. But so far there is a scarcity of chest x-ray image dataset. And covid-19 itself is a very new kind of disease where even scientists are struggling to figure out the nature of coronavirus, as it constantly mutates over the period of time. So considering this, it’s no wonder how tough it is to build a system that may detect covid patients through chest x-ray images.

# Project Objectives
In this project, 4 different models are developed that can detect covid infected chest x-ray images. These models are: ResNet18, ResNet50, SqueezeNet, and DenseNet121 by following a transfer learning based approach, used google colaboratory and pytorch while developing our models and followed the following procedure while developing models.

Data pre-processing:
In data preprocessing, resized the image, croped them, converted them into pytorch tensors, and normalized them using pre-computed mean and standard deviation.


Set the hyper-parameters: 
Then set the hyper-parameters for our task. Here epochs, batch size, learning rate and momentum are treated as hyper-parameters.

Setup the model: 
A total of four models are developed. For each of them, used the pre-trained model provided pytorch, and as a part of transfer-learning modified the last layer according to our purpose. Cross-entropy-loss as is used as the loss function and stochastic-gradient-descent (SGD) as optimizer. To decay the learning rate based on epoch number we used pytoch’s “lr_scheduler.StepLR” class.


Model training: 
Pytorch provides a ‘DataLoader’ class which makes iterating over datasets much easier. It breaks the dataset into chunks (batches) according to the given batch size. This step is very essential as the model is not trained with the whole dataset at once, but we train our model with one batch after another.


Accuracy measurements: 
After the training evaluated model’s performance based on the sensitivity, specificity, confusion matrix, and ROC curve. 
