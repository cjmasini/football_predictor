
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd

def readin():
    stat_file = open('./conference_stat.csv','r')
    next(stat_file, None)    #remove header
    stat=[]
    for line in stat_file:
        line = line.strip().split(",")[1:43]+line.strip().split(",")[-1:-25:-1]
        line = np.array(list(map(float,line)))    #only take the score part
        stat.append(line)

    result_file = open('./conference_res.csv','r')
    next(result_file, None)     #remove header

    results=[]

    for line in result_file:
        line = line.strip().split(",")
        line = line[-1:-3:-1]    #only take the score part
        line.reverse()
        line = np.array(list(map(float,line)))    #only take the score part
        results.append(line)

    stat, results = np.array(stat),np.array(results)

    #normalization
    stat = (stat-stat.min(axis=0))/(stat.max(axis=0)-stat.min(axis=0))-0.5
    #red_results = (red_results-red_results.min(axis=0))/(red_results.max(axis=0)-red_results.min(axis=0))-0.5

    X_train, X_test, Y_train, Y_test= stat[:-5],stat[-5:],results[:-5],results[-5:]
    return  X_train, X_test, Y_train, Y_test

class Network():

    def __init__(self, nn_param_choices):
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters
        self.prediction_compare = pd.DataFrame()
        self.best_network = {}
        self.best_score = 0



    def create_random(self):
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        self.network = network

    def compile_model(self,nb_classes,input_shape):
        nb_layers = self.network['nb_layers']
        nb_neurons = self.network['nb_neurons']
        activation = self.network['activation']
        optimizer = self.network['optimizer']
        model = Sequential()
        for i in range(nb_layers):
            if i == 0:
                model.add(Dense(nb_neurons, activation=activation, input_shape=(input_shape,)))
        else:
                model.add(Dense(nb_neurons, activation=activation))
        model.add(Dropout(0.2))  # hard-coded dropout
        #model.add(Dense(nb_classes, activation='softmax'))
        model.add(Dense(nb_classes))

        model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['accuracy'])

        return model

    def train_score(self, x_train, y_train, x_test, y_test):
        nb_classes = y_train.shape[1]
        input_shape = x_train.shape[1]
        model = self.compile_model(nb_classes,input_shape)
        model.fit(x_train, y_train,epochs=40,verbose=0)
        pred_train = model.predict(x_train)
        pred = model.predict(x_test)

        prediction_compare = pd.DataFrame()
        win_acc = 0.0
        score_diff = 0.0
        score_diff_sum = 0.0
        std = 0.0
        std_sum = 0.0
        avg_spread = 0.0
        vegas = pd.read_csv("../2012-2018_vegas_predictions.csv")
        for i in range(x_train.shape[0]):
            if (pred_train[i][0]-pred_train[i][1])*(y_train[i][0]-y_train[i][1])>0:
                win_acc+=1.0
            score_diff += 0.5*(abs(pred_train[i][0] - y_train[i][0]) + abs(pred_train[i][1] - y_train[i][1]))
            std += 0.5*(pred_train[i][0] - y_train[i][0])**2 + (pred_train[i][1] - y_train[i][1])**2

            score_diff_sum += abs(pred_train[i][0] - y_train[i][0] + pred_train[i][1] - y_train[i][1])
            std_sum += (pred_train[i][0] - y_train[i][0] + pred_train[i][1] - y_train[i][1])**2
            prediction_compare = prediction_compare.append(pd.DataFrame({'actual_1': y_train[i][0], 'actual_2': y_train[i][1], 'pred_1': pred_train[i][0], 'pred_2': pred_train[i][1], 'spread': vegas.iloc[i].spread, 'over_under': vegas.iloc[i].spread}, index=[i]))

        win_acc/=x_train.shape[0]
        score_diff /= x_train.shape[0]
        std /= x_train.shape[0]
        std = std**0.5

        score_diff_sum /= x_train.shape[0]
        std_sum /= x_train.shape[0]
        std_sum = std_sum**0.5




        print("win_acc = "+str(win_acc))
        print("score_diff = "+str(score_diff))
        print("std = "+str(std))
        print("score_diff_sum = " +str(score_diff_sum))
        print("std_sum = "+str(std_sum))

        # define accuracy score (fitness score):
        score = 0
        ps = ""
        for i in range(x_test.shape[0]):
            ps += "predicted: "+str(pred[i][0])+" vs "+str(pred[i][1])
            ps += "actual: "+str(y_test[i][0])+" vs "+str(y_test[i][1])+'\n'
            print("predicted: "+str(pred[i][0])+" vs "+str(pred[i][1]))
            print("actual: "+str(y_test[i][0])+" vs "+str(y_test[i][1]))
            pred_diff = (pred[i][0]-pred[i][1])/(pred[i][0]+pred[i][1])

            if(y_test[i][0]-y_test[i][1])==0:
                score += 1-abs(pred_diff)
            else:
                test_diff = (y_test[i][0]-y_test[i][1])/(y_test[i][0]+y_test[i][1])
                score += pred_diff/test_diff



        print(self.network)
        score =score/x_test.shape[0]
        if score > self.best_score:
            self.best_score = score
            self.best_network = self.network
            self.prediction_compare = prediction_compare
            print("Best network:",self.best_network)
            print("Best score:",score)
            print(ps)
            prediction_compare.to_csv("prediction_compare.csv")
        print("score = "+str(score))
        self.accuracy = score



        print()




# In[2]:


def create_population(count, nn_param_choices):
    pop = []
    for _ in range(0, count):
        network = Network(nn_param_choices)
        network.create_random()
        pop.append(network)
    return pop

def breed(nn_param_choices,mother,father,mutate_chance=0.2):
    children = []
    for _ in range(2): # two children
        child = {}
            # Loop through the parameters and pick params for the kid.
        for param in nn_param_choices:
            child[param] = random.choice([mother.network[param], father.network[param]])

        network = Network(nn_param_choices)
        network.create_set(child)

        if mutate_chance > random.random():
            network = mutate(nn_param_choices,network)

        children.append(network)

    return children

def mutate(nn_param_choices, network):
    mutation = random.choice(list(nn_param_choices.keys()))
    network.network[mutation] = random.choice(nn_param_choices[mutation])
    return network

def evolve(pop,nn_param_choices,retain=0.5,random_select_rate=0.1):
    graded = pop
        # Get the number we want to keep for the next gen.
    retain_length = int(len(graded)*retain)
        # The parents are every network we want to keep.
    parents = graded[:retain_length]
        # For those we aren't keeping, randomly keep some anyway.
    for individual in graded[retain_length:]:
        if random_select_rate > random.random():
            parents.append(individual)

        # Now find out how many spots we have left to fill.
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []

        # Add children, which are bred from two remaining networks.
    while len(children) < desired_length:
            # Get a random mom and dad.
        male = random.randint(0, parents_length-1)
        female = random.randint(0, parents_length-1)

            # Assuming they aren't the same network...
        if male != female:
            male = parents[male]
            female = parents[female]
            babies = breed(nn_param_choices,male, female)

            for baby in babies:
                    # Don't grow larger than desired length.
                if len(children) < desired_length:
                    children.append(baby)

    parents.extend(children)
    #for parent in parents:
        #print("parents are:")
        #print(parent.network)

    return parents


# In[3]:


def train_networks(networks, x_train, y_train, x_test, y_test):
    for network in networks:
        network.train_score(x_train, y_train, x_test, y_test)

def get_average_accuracy(networks):
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, x_train, y_train, x_test, y_test):
    fitness_history = []
    networks = create_population(population,nn_param_choices)
    # Evolve the generation.
    for i in range(generations):
        print("this is "+str(i)+"th generation:")
        train_networks(networks, x_train, y_train, x_test, y_test)
        average_accuracy = get_average_accuracy(networks)
        print("average_accuracy= "+str(average_accuracy))
        fitness_history.append(average_accuracy)
        if i != generations - 1:
            networks = evolve(networks,nn_param_choices)

    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)
    print("network rank:")
    for individual in networks:
        print(individual.network)
        print(individual.accuracy)
    return  fitness_history



# In[4]:


#print(np.array(fitness_history))


# In[ ]:


#Best_model = {{'nb_neurons': 128, 'nb_layers': 2, 'activation': 'relu', 'optimizer': 'rmsprop'}}

x_train, x_test, y_train, y_test = readin()

generations = 20  # Number of times to evole the population.
population = 20  # Number of networks in each generation.


nn_param_choices = {'nb_neurons': [16, 32, 64, 128],
                    'nb_layers': [1,2,3,4],
                    'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
                    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam'],}

fitness_history = generate(generations, population,
                           nn_param_choices,x_train, y_train, x_test, y_test)
plt.plot(fitness_history)
plt.xlabel("generations",fontsize=20)
plt.ylabel("fitness score",fontsize=20)
plt.show()
