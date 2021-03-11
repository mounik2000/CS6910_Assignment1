from keras.datasets import fashion_mnist
import numpy as np
import wandb

#Question 1
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
        

# Question 2
#dimensions : 
#X_input -> n
#a_i,h_i : (ith hidden_layer_size)
#W_list[i] : (ith hidden_layer_size) x ({i-1}th hidden_layer_size)
#b_list[i] : (ith hidden_layer_size)
#Output : h_list (h_1,h_2....h_{num_hidden_layers}), a_list (a_1,a_2....a_{num_hidden_layers+1})), y
def feed_forward_neural_network(X_input, W_list, b_list, activation_function, num_hidden_layers, hidden_layer_size) :
  
    X_input = X_input.reshape(-1,1)
    h_prev = X_input
    a_i = []
    h_list = []
    a_list = []
    #Other than o/p layer
    for i in range(0, num_hidden_layers) :
        a_i = b_list[i].reshape(-1,1) + np.dot(W_list[i], h_prev)
        a_list.append(a_i)
        if activation_function == 'sigmoid' :
            h_prev = 1 /( 1 + np.exp(-1 * a_i))
        elif activation_function == 'tanh' :
            h_prev = np.tanh(a_i)
        elif activation_function == 'ReLU' :
            h_prev = np.maximum(0, a_i)
        h_list.append(h_prev)
    a_i = b_list[num_hidden_layers].reshape(-1,1) + np.dot(W_list[num_hidden_layers], h_prev)    
    #soft_max
    a_list.append(a_i)
    a_i = np.exp(a_i)
    summ = np.sum((a_i))
    y = a_i / summ
    parameters = []
    parameters.append(h_list)
    parameters.append(a_list)
    parameters.append(y)
    return parameters
