import matplotlib.pyplot as plt
import numpy as np
from sklearn import utils
import math
import os
import cv2
from tqdm import tqdm
import json

'''
HYPERPARAMETER
'''
lr = 0.1
img_height = 240
img_width = 320
img_size = ((img_height,img_width))
batch_size = 100
epoch = 50


'''
LOAD AND SPLIT DATASET
'''
# Data Loader
def load_data(dataset_path,image_size,class_names,batch_size = 100):
    # List of data and label
    x = list()
    y = list()

    for class_name in class_names:
        class_path = os.path.join(dataset_path,class_name)
        file_list = [os.path.splitext(filename)[0] for filename in os.listdir(class_path)]
        x.append(file_list)
        y.append([class_name] * len(file_list))

    x = x[0][:batch_size] + x[1][:batch_size] + x[2][:batch_size]
    y = y[0][:batch_size] + y[1][:batch_size] + y[2][:batch_size]

    return x,y

'''
PREPARE DATASET
'''
dataset_path = (os.path.join("flowers"))
class_names = ['daisy','dandelion','sunflower']
x,y = load_data(dataset_path,img_size,class_names,batch_size)

# Split data into training and test dataset with ratio (80:20)
def split_data(data,train_size = 0.8):
    train_length = math.floor(train_size * len(data))
    train_data = np.array(data[:train_length])
    test_data = np.array(data[train_length:])
    return train_data,test_data

x,y = utils.shuffle(x,y) #shuffle to balancing data distribution
train_ds,test_ds = split_data(x)
train_val,test_val = split_data(y)
print("TRAIN DATA\t= {}\nTEST DATA\t= {}".format(train_ds.shape[0],test_ds.shape[0]))

'''
PREPROCESSING
'''
# Convolution and Maxpool Layer (if needed)
def conv2d(image):
    kernel = 1/9 * np.array([[1,1,1],
                            [1,1,1],
                            [1,1,1]])

    newImg = np.zeros((image.shape[0]-2,image.shape[1]-2))
   
    for i in range(image.shape[1]-2):
        for j in range(image.shape[0]-2):
            for x in range(len(kernel)):
                for y in range(len(kernel[0])):
                    newImg[j][i] += image[j+x][i+y] * kernel[x][y]

    return newImg.astype(np.uint8)

def maxpool2d(image):
    newImg = np.zeros((image.shape[0]//2+1,image.shape[1]//2+1))

    for i in range(0,image.shape[1],2):
        for j in range(0,image.shape[0],2):
            pool = image[j:j+2,i:i+2]
            newImg[j//2][i//2] = np.max(pool)
            
    return newImg.astype(np.uint8)

# Use architecture conv-max(2 times)
def convmaxpool(image):
    newImg = conv2d(image)
    newImg = maxpool2d(newImg)
    for i in range(1):
        newImg = conv2d(newImg)
        newImg = maxpool2d(newImg)
    return newImg

'''
PREPROCESSING
'''
def preprocess(ds,label):
    input_data = list()
    for i in tqdm(range(ds.shape[0]),desc="PREPROCESS ="):
        image_path = os.path.join("flowers", label[i], ds[i]+".jpg")
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,img_size,interpolation=cv2.INTER_AREA)

        # Convolution and Maxpooling Image
        image = convmaxpool(image)

        # Flatten 2D data to 1D data
        input_data.append(image.flatten())

    return input_data

train_ds = preprocess(train_ds,train_val)
test_ds = preprocess(test_ds,test_val)



'''
DEFINE MODEL
-> input image (320,240)
1. Conv2D (318,238)
   MaxPool2D (160,120)
   Conv2D (158,118)
   MaxPool2D (80,60)
2. Flatten (1200)
3. Layer 1 (100)
4. Layer 2 (50)
5. Output Layer (3)
'''

# Number of perceptron each hidden layers (use 2 hidden layers)
layer1 = 50
layer2 = 25

# Initialize Theta and Bias for each layers
theta1 = np.random.rand(layer1,80 * 60) # Output of convolution and maxpool
bias1 = np.random.rand(layer1)
theta2 = np.random.rand(layer2,layer1)
bias2 = np.random.rand(layer2)
theta_out = np.random.rand(3,layer2)
bias_out = np.random.rand(3)
total_param = theta1.flatten().shape[0] + bias1.flatten().shape[0] \
            + theta2.flatten().shape[0] + bias2.flatten().shape[0] \
            + theta_out.flatten().shape[0] + bias_out.flatten().shape[0]
print("TOTAL PARAMETER TRAINED =",total_param)

# Activation Function
def sigmoid_act(x):
    val = 1/(1+np.exp(-x))
    return val

def softmax_act(y):
    val = np.exp(y)/sum(np.exp(y))
    return val

# FeedForward Function
def feedForward(input_data):
    z1 = sigmoid_act(np.dot(theta1, input_data) + bias1) # output layer 1 
    z2 = sigmoid_act(np.dot(theta2, z1) + bias2) # output layer 2
    y = sigmoid_act(np.dot(theta_out, z2) + bias_out) # Output of the Output layer
    return (z1,z2,y)

# Backpropagation and Gradient Descent Function
def backPropagate(input_data,forwards,dOut):
    delta_Out = np.dot(np.expand_dims(forwards[1],axis=0).T,np.expand_dims(dOut,axis=0))
    delta_2 = np.dot(np.expand_dims(forwards[0],axis=0).T,np.expand_dims(np.dot(dOut,theta_out),axis=0))
    delta_1 = np.dot(np.expand_dims(input_data,axis=0).T,np.expand_dims(np.dot(forwards[1],theta2),axis=0))
    db_Out = dOut
    db_2 = np.dot(dOut,theta_out)
    db_1 = np.dot(forwards[1],theta2)
    return (delta_Out,delta_2,delta_1),(db_Out,db_2,db_1)


def gradientDescent(dthetas,dbias):
    global theta1,bias1,theta2,bias2,theta_out,bias_out
    theta_out = theta_out - lr*dthetas[0].T
    theta2 = theta2 - lr*dthetas[1].T
    theta1 = theta1 - lr*dthetas[2].T
    bias_out = bias_out - lr*dbias[0][0]
    bias2 = bias2 - lr*dbias[1][0]
    bias1 = bias1 - lr*dbias[1][0]

# Metrics Function
def sumSquareloss(y_predict,y_true):
    return (1/2) * pow((y_predict-y_true),2)

def crossEntropy(y_predict, y_true):
    loss = []
    for i in range(len(y_true)):
        if y[i] == 1:
            loss.append(-np.log(y_predict[i]))
        else:
            loss.append(-np.log(1 - y_predict[i]))
    return loss

# Metrics Container
loss = list()
error = 0

# Train Model
def train(epoch):
    for e in range(epoch):
        for i in tqdm(range(len(train_ds)),desc="EPOCH {} =".format(e)):
            # Input data from train dataset
            input_data = train_ds[i]

            # Feed Forward for Result each Nodes
            forwards = feedForward(input_data)

            # Error using SoftMax Activation
            y = softmax_act(forwards[-1])
            one_hot = list()

            for label in class_names:
                if label == train_val[i]:
                    one_hot.append(1)
                else:
                    one_hot.append(0)
                    
            # error = sumSquareloss(y,one_hot)
            error = crossEntropy(y,one_hot)

            # Compute Delta Theta
            dOut = 2 * (y-one_hot) * (1-y) * y            
    
            # Backpropagate
            dthetas,dbias = backPropagate(input_data,forwards,dOut)
            
            # Gradient Descent
            gradientDescent(dthetas,dbias)

        # Save Metric Value
        loss.append(error)

train(epoch)

'''
METRICS VISUALIZATION
'''
loss = np.array(loss)
# open output file for writing
with open('loss.txt', 'w') as filehandle:
    json.dump(loss.tolist(), filehandle)

# plt.plot(np.arange(epoch * train_ds.shape[0]),loss.T[0],label = 'daisy')
# plt.plot(np.arange(epoch * train_ds.shape[0]),loss.T[1],label = 'dandelion')
# plt.plot(np.arange(epoch * train_ds.shape[0]),loss.T[2],label = 'sunflower')
# plt.xlabel("Iterate")
# plt.ylabel("Cost")
# plt.legend()
# plt.show()


'''
EVALUATION FOR ACCURACY
'''
# Evaluate model with test dataset
def evaluate(test_ds,test_val):
    accuracy = 0
    for i in range(len(test_ds)):
        # Input data from test dataset
        input_data = test_ds[i]

        # Feed Forward for Result each Nodes
        forwards = feedForward(input_data)

        # Error using SoftMax Activation
        y = softmax_act(forwards[2])
        class_name = class_names[np.argmax(y)]
        if class_name == test_val[i]:
            accuracy +=1

    print("Accuracy = {:.2f}%".format((accuracy/len(test_ds)) * 100))
    return accuracy/len(test_ds)

evaluate(test_ds,test_val)


'''
PREDICT FUNCTION
'''
def predict(input_path):
    image = cv2.imread(input_path,cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,img_size,interpolation=cv2.INTER_AREA)

    # Convolution and Maxpooling Image
    image = convmaxpool(image)

    # Flatten 2D data to 1D data
    input_data = image.flatten()

    # Feed Forward for Result each Nodes
    forwards = feedForward(input_data)

    y = softmax_act(forwards[-1])
    print("Predicted as {} with {:.2f}% of score".format(class_names[np.argmax(y)],np.max(forwards[2]) * 100))

predict('flowers/daisy/5547758_eea9edfd54_n.jpg')