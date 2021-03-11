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


# Question 3

# Initializing weights and biases
def initialized_weights(weight_initializer,num_hidden_layers,hidden_layer_size):
  if weight_initializer == 'random' :
    L = num_hidden_layers+1
    K = hidden_layer_size
    W_list = []
    #Error 2??
    W_list.append(np.random.randint(-1,1,size=(K,784)))
    for i in range(0, num_hidden_layers-1) :
      W_list.append(np.random.randint(-1,1,size=(K,K)))
    W_list.append(np.random.randint(-1,1,size=(10,K)))
    b_list = []
    for i in range(0,num_hidden_layers) :
      b_list.append(np.random.randint(-1,1,size=(K,1)))
    b_list.append(np.random.randint(-1,1,size=(10,1)))
    W_list = np.array(W_list)
    b_list = np.array(b_list)
    return [W_list, b_list]
  if weight_initializer == 'Xavier' :
    L = num_hidden_layers+1
    K = hidden_layer_size
    W_list = []
    b_list = []
    W_list.append(np.random.uniform(-1.0/28,1.0/28,[K,784]))
    for i in range(0, num_hidden_layers-1) :
      W_list.append(np.random.uniform(-1.0/np.sqrt(K),1.0/np.sqrt(K),[K,K]))
    W_list.append(np.random.uniform(-1.0/np.sqrt(K),1.0/np.sqrt(K),[10,K]))
    for i in range(0,num_hidden_layers) :
      b_list.append(np.random.uniform(-1,1,(K,1)))
    b_list.append(np.random.uniform(-1,1,(10,1)))
    W_list = np.array(W_list)
    b_list = np.array(b_list)
    return [W_list, b_list]

# Compute ouput and loss
def get_loss_value_and_prediction(X,Y,weights,biases,activation_function,num_hidden_layers,hidden_layer_size, loss_type):
  y_pred = []
  for i in range(X.shape[0]):
    params = feed_forward_neural_network(X[i],weights,biases,activation_function,num_hidden_layers,hidden_layer_size)
    y = params[2]
    y_pred.append(y)
  y_pred = np.array(y_pred)
  Loss = 0
  num = X.shape[0]
  forming_Y = []
  for i in range(num):
    e_y = np.zeros((10,1))
    e_y[Y[i]] = 1
    forming_Y.append(e_y)
  if (loss_type == 'entropy'):
    for i in range(num):
      for j in range(10):
        if(forming_Y[i][j] == 1):
          Loss+=(1/num)*(-1*np.log(y_pred[i][j]))
  else:
    for i in range(num):
      for j in range(10):
        Loss+= (forming_Y[i][j]-y_pred[i][j])**2
  return [Loss,y_pred]

# Initialize dw,db to 0
def initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size ) :
  K = hidden_layer_size
  #dw
  diff_W_list = []
  diff_W_list.append(np.zeros((K,784)))
  for i in range(0, num_hidden_layers-1) :
    diff_W_list.append(np.zeros((K, K)))
  diff_W_list.append(np.zeros((10, K))) 
  #db
  diff_b_list = []
  for i in range(0,num_hidden_layers) :
    diff_b_list.append(np.zeros((K,1)))
  diff_b_list.append(np.zeros((10,1)))
  W_list = np.array(diff_W_list)
  b_list = np.array(diff_b_list)
  return [W_list, b_list]

# Get g' of a vector
def get_diff_g(h_k, activation_function):
  if activation_function == 'sigmoid' :
      # g'(z) = g(z) * (1 - g(z))
    for i in range(0, h_k.shape[0]) :
      h_k[i] = h_k[i] * (1 - h_k[i])

  elif activation_function == 'tanh' :
      # g'(z) = 1 - (g(z))^2
    for i in range(0, h_k.shape[0]) :
      h_k[i] = 1 - pow(h_k[i], 2)

  elif activation_function == 'ReLU' :
      # g'(z) = 1 if z is positive, 0 otherwise
    for i in range(0, h_k.shape[0]) :
      if (h_k[i]>0):
        h_k[i] = 1.0
      else:
        h_k[i] = 0.0
  return (h_k.reshape(-1, 1))   


#Find gradient of loss function w.r.t weights, biases for one input output
def back_propagation(W_list,b_list,x_input,y_true,loss_type,num_hidden_layers, hidden_layer_size,activation_function, parameters):
  L = num_hidden_layers+1
  h_list = parameters[0]
  a_list = parameters[1]
  y_pred = parameters[2]
  y_pred.reshape(-1,1)
  x_input = (x_input).reshape(-1,1)
  e_y = np.zeros((10,1))
  e_y[y_true] = [1]
  diff_a_list = [] # L
  diff_h_list = [] # L-1
  diff_W_list = [] # L
  diff_b_list = [] # L
  for i in range(0, L) :
    diff_a_list.append(0)
    diff_h_list.append(0)
    diff_W_list.append(0)
    diff_b_list.append(0)
  diff_h_list.pop(0) # to make size L-1
  if (loss_type == 'entropy'):
    diff_a_list[L-1] =  - ( e_y - y_pred ) 
  else:
    #Filling square loss
    diff_a_list[L-1] = []
    for j in range(10):
      #Derivative w.r.t a[L-1][j]
      p = 0
      for i in range(10):
        if (i == j):
          p += 2*(y_pred[j]-e_y[j])*(y_pred[j]-y_pred[j]*y_pred[j])
        else:
          p += 2*(y_pred[i]-e_y[i])*(-y_pred[i]*y_pred[j])
      diff_a_list[L-1].append(p)
    diff_a_list[L-1] = np.array(diff_a_list[L-1])
  for k in range(L-1, 0, -1) :
    diff_W_list[k] = np.dot(diff_a_list[k],(h_list[k-1]).T)
    diff_b_list[k] = diff_a_list[k]
    diff_h_list[k-1] = np.dot(W_list[k].T, diff_a_list[k])
    diff_a_list[k-1] = np.multiply ( diff_h_list[k-1], get_diff_g(h_list[k-1], activation_function))
  diff_W_list[0] = np.dot(diff_a_list[0], x_input.T)
  diff_b_list[0] = diff_a_list[0]
  return [np.array(diff_W_list), np.array(diff_b_list)]


#Training and Validation
def split_train_valid_data(X_train,Y_train):
    if(len(X_train) == 0):
        A = np.array([])
        return [A,A,A,A]
    size = len(X_train)
    arr = np.arange(size)
    np.random.shuffle(arr)
    train_X = []
    train_Y = []
    valid_X = []
    valid_Y = []
    for i in range(size):
      if (10*i<size):
        valid_X.append(X_train[i])
        valid_Y.append(Y_train[i])
      else:
        train_X.append(X_train[i])
        train_Y.append(Y_train[i])
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    valid_X = np.array(valid_X)
    valid_Y = np.array(valid_Y)
    return [train_X,train_Y,valid_X,valid_Y]


# Apply gradient descent
def updated_weights_gd(train_X, train_Y, hyper_parameter_combination,initial_weights,initial_biases,loss_type):
  num_epochs = hyper_parameter_combination["num_epochs"]
  num_hidden_layers = hyper_parameter_combination["num_hidden_layers"]
  hidden_layer_size = hyper_parameter_combination["hidden_layer_size"]
  optimizer = hyper_parameter_combination["optimizer"]
  learning_rate = hyper_parameter_combination["learning_rate"]
  batch_size = hyper_parameter_combination["batch_size"]
  weight_decay = hyper_parameter_combination["weight_decay"]
  weight_initializer = hyper_parameter_combination["weight_initializer"]
  activation_function = hyper_parameter_combination["activation_function"]
  if (optimizer == 'vanilla'):
    return do_sgd(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,len(train_X))
  elif (optimizer == 'sgd'):
    return do_sgd(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,1)
  elif (optimizer == 'minibatch'):
    return do_sgd(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,batch_size)
  elif (optimizer == 'momentum'):
    return do_momentum(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination)
  elif (optimizer == 'nesterov'):
    return do_nesterov(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination)
  elif (optimizer == 'rmsprop'):
    return do_rmsprop(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination)
  elif (optimizer == 'adam'):
    return do_adam(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination)    
  elif (optimizer == 'nadam'):
    return do_nadam(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination)
  else:
    return do_sgd(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,batch_size)    

#Backpropagation code
def feedforward_with_backpropagation(train_X, train_Y, valid_X,valid_Y, hyper_parameter_combination,loss_type):
  num_epochs = hyper_parameter_combination["num_epochs"]
  num_hidden_layers = hyper_parameter_combination["num_hidden_layers"]
  hidden_layer_size = hyper_parameter_combination["hidden_layer_size"]
  optimizer = hyper_parameter_combination["optimizer"]
  learning_rate = hyper_parameter_combination["learning_rate"]
  batch_size = hyper_parameter_combination["batch_size"]
  weight_decay = hyper_parameter_combination["weight_decay"]
  weight_initializer = hyper_parameter_combination["weight_initializer"]
  activation_function = hyper_parameter_combination["activation_function"]
  #initialize weights
  [initial_weights,initial_biases] = initialized_weights(weight_initializer,num_hidden_layers,hidden_layer_size)
  #use gd and update weights
  [final_weights,final_biases] = updated_weights_gd(train_X, train_Y, hyper_parameter_combination,initial_weights,initial_biases,loss_type)
  #find o/p after updating weights
  return [initial_weights,initial_biases,final_weights,final_biases]

#get accuracy
def get_accuracy(Y_true,Y_pred):
  size = len(Y_true)
  count = 0
  for i in range(size):
    print(Y_true[i])
    print(Y_pred[i])
    max_i = 0
    max_j = 0
    for j in range(10):
      if(Y_pred[i][j]>max_i):
        max_j = j
        max_i = Y_pred[i][j]
    if (Y_true[i] == max_j):
      count+=1
  return count/size
