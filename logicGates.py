from NeuralNetwork import NeuralNetwork
import torch
import numpy as np

class AND:
    def __init__(self):
        self.gate = NeuralNetwork([2, 1])
        #self.theta = self.gate.getLayer(0)
        #self.theta.fill_(0)
        #print self.theta
        #self.theta += torch.FloatTensor([[-10], [6], [6]])

    def __call__(self, input_1, input_2):
        if input_1 == True:
            self.input_1 = 1
        if input_1 == False:
            self.input_1 = 0
        if input_2 == True:
            self.input_2 = 1
        if input_2 == False:
            self.input_2 = 0

        result = np.around(self.forward().numpy()) #round output of propagation to 0 or 1
        #print result
        if result == 1:
            return True
        else:
            return False

    def forward(self):
        return self.gate.forward(torch.FloatTensor([[self.input_1,self.input_2]]))

    def train(self):
        #print 'In Train'
        data = torch.FloatTensor([[1,1],[1,0], [0,1], [0,0]])
        index = torch.randperm(4)
        train_data = torch.index_select(data, 0, index)
        #print train_data
        #print data

        #for i in range(100):
        while self.gate.total_error_neural_network > 0.001:
            output_for_train = torch.FloatTensor(torch.rand(len(data)))
            output_for_train = torch.unsqueeze(output_for_train, 1)

            for i in range(len(output_for_train)):
                if train_data[i, 0] == 1 and train_data[i, 1] == 1:
                    output_for_train[i] = 1.0
                elif train_data[i, 0] == 1 and train_data[i, 1] == 0:
                    output_for_train[i] = 0.0
                elif train_data[i, 0] == 0 and train_data[i, 1] == 1:
                    output_for_train[i] = 0.0
                else:
                    output_for_train[i] = 0.0
            #print output_for_train

            output_feedforward = self.gate.forward(train_data)
            #print "output_feedforward: " +str(output)
            self.gate.backward(output_for_train)
            self.gate.updateParams(0.1)
            #print "Neural Network Error:\t"+str(self.gate.total_error_neural_network)
        print "Hand Crafted Weights for HW02: \t" + str(torch.FloatTensor([[-10], [6], [6]]))
        print "Evaluated Weights for AND: \t"+str(self.gate.getLayer(0))

class OR:
    def __init__(self):
        self.gate = NeuralNetwork([2, 1])
        # self.theta = self.gate.getLayer(0)
        # self.theta.fill_(0)
        # self.theta += torch.FloatTensor([[-1], [2], [2]])

    def __call__(self, input_1, input_2):
        if input_1 == True:
            self.input_1 = 1
        if input_1 == False:
            self.input_1 = 0
        if input_2 == True:
            self.input_2 = 1
        if input_2 == False:
            self.input_2 = 0

        result = np.around(self.forward().numpy()) #round output of propagation to 0 or 1
        #print result
        if result == 1:
            return True
        else:
            return False

    def forward(self):
        return self.gate.forward(torch.FloatTensor([[self.input_1,self.input_2]]))

    def train(self):
        #print 'In Train'
        data = torch.FloatTensor([[1,1],[1,0], [0,1], [0,0]])
        index = torch.randperm(4)
        train_data = torch.index_select(data, 0, index)
        #print train_data
        #print data

        #for i in range(100):
        while self.gate.total_error_neural_network > 0.001:
            output_for_train = torch.FloatTensor(torch.rand(len(data)))
            output_for_train = torch.unsqueeze(output_for_train, 1)

            for i in range(len(output_for_train)):
                if train_data[i, 0] == 1 and train_data[i, 1] == 1:
                    output_for_train[i] = 1.0
                elif train_data[i, 0] == 1 and train_data[i, 1] == 0:
                    output_for_train[i] = 1.0
                elif train_data[i, 0] == 0 and train_data[i, 1] == 1:
                    output_for_train[i] = 1.0
                else:
                    output_for_train[i] = 0.0
            #print output_for_train

            output_feedforward = self.gate.forward(train_data)
            #print "output_feedforward: " +str(output)
            self.gate.backward(output_for_train)
            self.gate.updateParams(0.1)
            #print "Neural Network Error:\t"+str(self.gate.total_error_neural_network)
        print "Hand Crafted Weights for HW02: \t" + str(torch.FloatTensor([[-1], [2], [2]]))
        print "Evaluated Weights for OR: \t"+str(self.gate.getLayer(0))


class NOT:
    def __init__(self):
        self.gate = NeuralNetwork([1, 1])
        # self.theta = self.gate.getLayer(0)
        # self.theta.fill_(0)
        # #print self.theta
        # self.theta += torch.FloatTensor([[1], [-2]])

    def __call__(self, input_1):
        if input_1 == True:
            self.input_1 = 1
        if input_1 == False:
            self.input_1 = 0

        result = np.around(self.forward().numpy()) #round output of propagation to 0 or 1
        #print result
        if result == 1:
            return True
        else:
            return False

    def forward(self):
        return self.gate.forward(torch.FloatTensor([[self.input_1]]))

    def train(self):
        #print 'In Train'
        data = torch.FloatTensor([[1],[0]])
        index = torch.randperm(2)
        train_data = torch.index_select(data, 0, index)
        #print train_data
        #print data

        #for i in range(100):
        while self.gate.total_error_neural_network > 0.001:
            output_for_train = torch.FloatTensor(torch.rand(len(data)))
            output_for_train = torch.unsqueeze(output_for_train, 1)

            for i in range(len(output_for_train)):
                if train_data[i, 0] == 1:
                    output_for_train[i] = 0.0
                else:
                    output_for_train[i] = 1.0

            #print output_for_train

            output_feedforward = self.gate.forward(train_data)
            #print "output_feedforward: " +str(output)
            self.gate.backward(output_for_train)
            self.gate.updateParams(0.1)
            #print "Neural Network Error:\t"+str(self.gate.total_error_neural_network)
        print "Hand Crafted Weights for HW02: \t" + str(torch.FloatTensor([[1], [-2]]))
        print "Evaluated Weights for NOT: \t"+str(self.gate.getLayer(0))

class XOR:
    def __init__(self):
        self.gate = NeuralNetwork([2, 2, 1])
        # self.theta_layer_0 = self.gate.getLayer(0)
        # self.theta_layer_0.fill_(0)
        # self.theta_layer_0 += torch.FloatTensor([[-25, -25], [-50, 50], [50, -50]])
        #
        # self.theta_layer_1 = self.gate.getLayer(1)
        # self.theta_layer_1.fill_(0)
        # self.theta_layer_1 += torch.FloatTensor([[-25], [50], [50]])

    def __call__(self, input_1, input_2):
        if input_1 == True:
            self.input_1 = 1
        if input_1 == False:
            self.input_1 = 0
        if input_2 == True:
            self.input_2 = 1
        if input_2 == False:
            self.input_2 = 0

        result = np.around(self.forward().numpy()) #round output of propagation to 0 or 1
        #print result
        if result == 1:
            return True
        else:
            return False

    def forward(self):
        return self.gate.forward(torch.FloatTensor([[self.input_1,self.input_2]]))

    def train(self):
        #print 'In Train'
        data = torch.FloatTensor([[1,1],[1,0], [0,1], [0,0]])
        index = torch.randperm(4)
        train_data = torch.index_select(data, 0, index)
        #print train_data
        #print data

        #for i in range(100):
        while self.gate.total_error_neural_network > 0.001:
            output_for_train = torch.FloatTensor(torch.rand(len(data)))
            output_for_train = torch.unsqueeze(output_for_train, 1)

            for i in range(len(output_for_train)):
                if train_data[i, 0] == 1 and train_data[i, 1] == 1:
                    output_for_train[i] = 0.0
                elif train_data[i, 0] == 1 and train_data[i, 1] == 0:
                    output_for_train[i] = 1.0
                elif train_data[i, 0] == 0 and train_data[i, 1] == 1:
                    output_for_train[i] = 1.0
                else:
                    output_for_train[i] = 0.0
            #print output_for_train

            output_feedforward = self.gate.forward(train_data)
            #print "output_feedforward: " +str(output)
            self.gate.backward(output_for_train)
            self.gate.updateParams(3.0)
            #print "Neural Network Error:\t"+str(self.gate.total_error_neural_network)
        print "Hand Crafted Weights for HW02 (Layer_0): \t" + str(torch.FloatTensor([[-25, -25], [-50, 50], [50, -50]]))
        print "Hand Crafted Weights for HW02 (Layer 1): \t" + str(torch.FloatTensor([[-25], [50], [50]]))

        print "Evaluated Weights for XOR (Layer_0): \t"+str(self.gate.getLayer(0))
        print "Evaluated Weights for XOR (Layer_1): \t"+str(self.gate.getLayer(1))
