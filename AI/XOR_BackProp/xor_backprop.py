import math

from typing import Dict, List, Tuple
from random import randint
from functools import reduce

class Perceptron(object):
    def __init__(self, input_number: int, bias_input: List[float] = [], input_weights: List[float] = []):
        self.input_weight: List[float] = input_weights
        if not self.input_weight:
            self.input_weight = [
                randint(-50, 50)*.01 for i in range(input_number+1)]
        # self.input_weight:List[float]=[-.3,.21,.15]
        self.bias_input: List[float] = bias_input
        self.output: float = 0
        self.delta: float = 0

    def step_function(self, agg_input):
        return 1/(1+math.exp(-agg_input))

    def train(self, inputs: List[float]):
        input_weight = zip(self.input_weight, self.bias_input+inputs)
        agg_input = reduce(lambda x, y: (1, x[0]*x[1]+y[0]*y[1]), input_weight)
        self.output = self.step_function(agg_input[1])
    
    def test(self,inputs:List[float]):
        input_weight = zip(self.input_weight, self.bias_input+inputs)
        agg_input = reduce(lambda x, y: (1, x[0]*x[1]+y[0]*y[1]), input_weight)
        return self.step_function(agg_input[1])

# test = Perceptron(3, [1], [-.3, .21, .15])
class XOR_neural_net(object):
    def __init__(self):
        self.input_layer = [Perceptron(3, [1]),
                    Perceptron(3, [1])]
        self.output_layer = [Perceptron(3, [1])]
        self.train_input: List[Tuple[List[float], List[float]]] = [
             ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ]
    
    def train(self,epoc:int):
        for i in range(epoc):
            for train_dat in self.train_input:
                self.forward_propagation(train_dat[0])
                self.back_propagation(train_dat[0],train_dat[1])
    def test(self,inputs:List[float])->List[float]:
        #input_layer_output: List[float] = []
        input_layer_output: List[float] = []
        for input_percept in self.input_layer:
            input_layer_output.append(input_percept.test(inputs))

        # train output layer
        output_layer_output: List[float] = []
        for output_percept in self.output_layer:
            output_layer_output.append(output_percept.test(input_layer_output))
        return output_layer_output

    def forward_propagation(self,inputs:List[float]):
        # train input layer
        input_layer_output: List[float] = []
        for input_percept in self.input_layer:
            input_percept.train(inputs)
            input_layer_output.append(input_percept.output)

        # train output layer
        for output_percept in self.output_layer:
            output_percept.train(input_layer_output)

    def back_propagation(self,inputs:List[float],target_output:List[float]):
        # Ouput layer back propagation preparation
        # delta_output_layer = []
        for index_output_layer,output_percept in enumerate(self.output_layer):
            out_diff = target_output[index_output_layer]-output_percept.output
            func_output = output_percept.output
            output_percept.delta = out_diff*func_output*(1-func_output)
     
        # Input layer back propagation preparation
        for input_index, input_percept in enumerate(self.input_layer):
            delta_input_agg = 0.0
            for output_index, output_percept in enumerate(self.output_layer):
                delta_input_agg = delta_input_agg + output_percept.delta * \
                    output_percept.input_weight[input_index +
                                                len(output_percept.bias_input)]
            func_input = input_percept.output
            input_percept.delta = delta_input_agg*func_input*(1-func_input)
            

        ##################################
        # update the Weight of the persceptron
        
        # output layer weight update
        self.update_weights(input_percept, inputs)

    def update_weights(self, input_percept, inputs):
        # output layer weight update
        alpha = 1
        for output_percept in self.output_layer:
            input_list = output_percept.bias_input + \
                [input_percept.output for input_percept in self.input_layer]
            new_weight=map(lambda x: x[1]+alpha*x[0]*output_percept.delta,
                zip(input_list, output_percept.input_weight))
            output_percept.input_weight=list(new_weight)
            

        #input layer weight update
        for input_percept in self.input_layer:
            input_list = input_percept.bias_input + \
                inputs
            new_weight=map(lambda x: x[1]+alpha*x[0]*input_percept.delta,
                zip(input_list, input_percept.input_weight))
            input_percept.input_weight=list(new_weight)
            
    def output_weight(self):
        print("weights\nInput layer")
        for input_l in self.input_layer:
            print(input_l.input_weight)
        print("Output layer")
        for output_l in self.output_layer:
            print(output_l.input_weight)

xor_obj=XOR_neural_net()
xor_obj.output_weight()
xor_obj.train(10000)
print("Final weights")
xor_obj.output_weight()
print(f"input: 0,0 ouput {xor_obj.test([0,0])}")
print(f"input: 0,1 ouput {xor_obj.test([0,1])}")
print(f"input: 1,0 ouput {xor_obj.test([1,0])}")
print(f"input: 1,1 ouput {xor_obj.test([1,1])}")
# print(output_layer[0].output)
# test.train([0,0])
# print(test.output)
