# To run the program
# python3 /path/to/file/and_adline.py
import math

from typing import Dict, List, Tuple
from random import randint
from functools import reduce

class Adline(object):
    def __init__(self, input_number: int, bias_input: List[float] = [], input_weights: List[float] = []):
        self.input_weight: List[float] = input_weights
        if not self.input_weight:
            self.input_weight = [
                randint(-50, 50)*.01 for i in range(input_number+1)]
        # self.input_weight:List[float]=[-.3,.21,.15]
        self.bias_input: List[float] = bias_input
        self.output: float = 0
        self.delta: float = 0
        
    
    def train(self, inputs: List[float],target_output:float):
        input_weight = zip(self.input_weight, self.bias_input+inputs)
        agg_output = reduce(lambda x, y: (1, x[0]*x[1]+y[0]*y[1]), input_weight)
        self.output=agg_output[1]
        self.delta=target_output-self.output
    
    def test(self,inputs:List[float]):
        input_weight = zip(self.input_weight, self.bias_input+inputs)
        agg_input = reduce(lambda x, y: (1, x[0]*x[1]+y[0]*y[1]), input_weight)
        return (1 if agg_input[1]>=0.0 else -1)

class AND_network(object):
    def __init__(self):
        self.adline = Adline(2,[1])
        self.train_input: List[Tuple[List[float], List[float]]] = [
             ([-1, -1], -1),
            ([-1, 1], -1),
            ([1, -1], -1),
            ([1, 1], 1),
        ]

    def train(self,epoc:int):
        for i in range(epoc):
            for train_dat in self.train_input:
                self.adline.train(train_dat[0],train_dat[1])
                self.update_weight(train_dat[0])
                # print(f"{train_dat}\t output:{self.adline.output}\t delta:{self.adline.delta}")
            # print("\n")
    
    def update_weight(self,input:List[float]):
        alpha=.1
        weight_input_pair=zip(self.adline.input_weight,self.adline.bias_input+input)
        updated_weight_pair:List[float]=[]
        for key,(weight,input_all) in enumerate(weight_input_pair):
            updated_weight_pair.append(weight+alpha*input_all*self.adline.delta)
        # updated_weight_pair=list(map(lambda x: x[0]+x[1]*alpha*self.adline.delta,weight_input_pair))
        self.adline.input_weight=updated_weight_pair

and_obj=AND_network()
and_obj.train(10)
print(f"Input -1: -1 , Output: {and_obj.adline.test([-1,-1])}")
print(f"Input -1:  1 , Output: {and_obj.adline.test([-1,1])}")
print(f"Input  1: -1 , Output: {and_obj.adline.test([1,-1])}")
print(f"Input  1:  1 , Output: {and_obj.adline.test([1,1])}")