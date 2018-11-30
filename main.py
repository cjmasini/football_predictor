import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from readin import readin
import numpy as np

class Network():

    def __init__(self, nn_param_choices):        
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        etwork = network
        
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
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['accuracy'])
        return model
    
    def train_score(self, x_train, y_train, x_test, y_test):        
        nb_classes = y_train.shape[1]
        input_shape = x_train.shape[1]
        model = self.compile_model(nb_classes,input_shape)
        model.fit(x_train, y_train,epochs=10)                    
        pred = model.predict(x_test)
        #print(y_test)
        #print(pred)
        
        # define accuracy score (fitness score):
        score = 0
        for i in range(x_test.shape[0]):
            pred_diff = (pred[i][0]-pred[i][1])/(pred[i][0]+pred[i][1])
            
            if(y_test[i][0]-y_test[i][1])==0:
                score += 1-abs(pred_diff)
                 
            else:    
                test_diff = (y_test[i][0]-y_test[i][1])/(y_test[i][0]+y_test[i][1])
                score += pred_diff/test_diff 
              
        print(self.network)
        score =score/x_test.shape[0]
        self.accuracy = score
        print("score = "+str(score))

def create_population(count, nn_param_choices):
    pop = []
    for _ in range(0, count):
        network = Network(nn_param_choices)
        network.create_random()
        pop.append(network)
    return pop

def breed(nn_param_choices,mother,father,mutate_chance=0.3):
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

def evolve(pop,nn_param_choices,retain=0.4,random_select_rate=0.1):
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
    for parent in parents:
        print(parent.network)

    return parents


def train_networks(networks, x_train, y_train, x_test, y_test):
    for network in networks:
        network.train_score(x_train, y_train, x_test, y_test)
        
def get_average_accuracy(networks):
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, x_train, y_train, x_test, y_test):
    networks = create_population(population,nn_param_choices)
    # Evolve the generation.
    for i in range(generations):        
        train_networks(networks, x_train, y_train, x_test, y_test)
        average_accuracy = get_average_accuracy(networks)        
        if i != generations - 1:
            networks = evolve(networks,nn_param_choices)
            
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)
            
def main():
    x_train, x_test, y_train, y_test = readin()
    generations = 10  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.
    dataset = 'cifar10'
    nn_param_choices = {'nb_neurons': [16, 32, 64, 128],'nb_layers': [1,1,1,1],
                        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
                        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam'],}

    generate(generations, population, nn_param_choices,x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()
