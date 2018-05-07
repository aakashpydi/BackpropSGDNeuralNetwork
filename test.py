from NeuralNetwork import NeuralNetwork
from logicGates import AND
from logicGates import OR
from logicGates import NOT
from logicGates import XOR
import torch

print "Testing AND gate implementation."
_and = AND()
_and.train()
print "AND Truth Table"
print "INPUT_1\tINPUT_2\tAND_RESULT"
print "True\tTrue\t"+str(_and(True, True))
print "True\tFalse\t"+str(_and(True, False))
print "False\tTrue\t"+str(_and(False, True))
print "False\tFalse\t"+str(_and(False, False))


print "\nTesting OR gate implementation."
_or = OR()
_or.train()
print "OR Truth Table"
print "INPUT_1\tINPUT_2\tOR_RESULT"
print "True\tTrue\t"+str(_or(True, True))
print "True\tFalse\t"+str(_or(True, False))
print "False\tTrue\t"+str(_or(False, True))
print "False\tFalse\t"+str(_or(False, False))

print "\nTesting NOT gate implementation."
_not = NOT()
_not.train()
print "NOT Truth Table"
print "INPUT_1\tNOT_RESULT"
print "True\t"+str(_not(True))
print "False\t"+str(_not(False))

print "\nTesting XOR gate implementation."
_xor = XOR()
_xor.train()
print "XOR Truth Table"
print "INPUT_1\tINPUT_2\tXOR_RESULT"
print "True\tTrue\t"+str(_xor(True, True))
print "True\tFalse\t"+str(_xor(True, False))
print "False\tTrue\t"+str(_xor(False, True))
print "False\tFalse\t"+str(_xor(False, False))
