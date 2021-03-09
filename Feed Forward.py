from keras.datasets import fashion_mnist
import numpy as np
import wandb


def plot_sample_images():
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    classes = []
    classes.append("T-shirt/top")
    classes.extend(['Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
    print(X_train.shape)
    wandb.init(project = "Plot Images For Each Class")
    Images = []
    visited = set()
    for i in range(Y_train.shape[0]):
        class_num = Y_train[i]
        if(not(class_num in visited)):
            Images.append(wandb.Image(X_train[i], caption=classes[class_num]))
            visited.add(class_num)
        if(len(visited) == 10):
            break
    wandb.log({"Examples for each class": Images})
    
    
plot_sample_images()