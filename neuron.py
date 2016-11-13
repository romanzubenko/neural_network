from add_unit import AddUnit
from sigmoid import SigmoidUnit
from muliply_unit import MultiplyUnit
from weight import Weight

class Neuron:
	def __init__(self, neurons):
		
		self.neurons = neurons
		self.weights = []
		self.mults = []

		for neuron in self.neurons:
			weight = Weight()	
			mult = MultiplyUnit(neuron, weight)	
			
			self.weights.append(weight)
			self.mults.append(mult)

		self.add = AddUnit(self.mults)
		self.sigmoid = SigmoidUnit(self.add)

		self.value = 0
		self.grad = 0

	def forward(self):
		for weight in self.weights:
			weight.forward()

		for mult in self.mults:
			mult.forward()

		self.add.forward()
		self.sigmoid.forward()
		self.value = self.sigmoid.value
		self.grad = 0

	def backward(self):
		self.sigmoid.grad = self.grad
		self.sigmoid.backward()
		self.add.backward()

		for mult in self.mults:
			mult.backward()		

	def updateParams(self, learning_rate): 
		for weight in self.weights:
			weight.updateParam(learning_rate)