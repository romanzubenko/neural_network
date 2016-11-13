import math
import random

LEARNING_RATE = 0.05

class MSEUnit:
	def __init__(self, unit1, targetValue):
		self.u1 = unit1
		self.targetValue = targetValue
		self.value = 0
		self.grad = 1

	def forward(self):
		self.grad = 0
		self.value = math.pow((self.u1.value - self.targetValue), 2)

	def backward(self):
		self.grad = (2.0 * (self.u1.value - self.targetValue))
		self.u1.grad += self.grad

class AddUnit:
	def __init__(self, units):
		self.units = units
		self.value = 0
		self.grad = 0

	def forward(self):
		self.grad = 0
		self.value = 0
		for unit in self.units:
			self.value += unit.value
		

	def backward(self):
		for unit in self.units:
			unit.grad += self.grad

class MultiplyUnit:
	def __init__(self, unit1, unit2):
		self.u1 = unit1
		self.u2 = unit2
		self.value = 0
		self.grad = 0

	def forward(self):
		self.grad = 0
		self.value = self.u1.value * self.u2.value

	def backward(self):
		self.u1.grad += self.u2.value * self.grad
		self.u2.grad += self.u1.value * self.grad

class SigmoidUnit:
	def __init__(self, unit1):
		self.u1 = unit1
		self.value = 0
		self.grad = 0

	def forward(self):
		self.grad = 0
		self.value = 1.0 / (1.0 + math.exp(-self.u1.value))

	def backward(self):
		self.u1.grad += self.value * (1.0 - self.value) * self.grad

class Weight:
	def __init__(self):
		self.learning_step = LEARNING_RATE
		self.value = (random.random() * 2) - 1
		self.grad = 0

		print '\tinit weight..', self.value

	def forward(self):
		self.grad = 0
		self.value = self.value

	def updateParam(self,):
		self.value += (-1) * self.grad * self.learning_step

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

	def updateParams(self): 
		for weight in self.weights:
			weight.updateParam()


class InputUnit:
	def __init__(self, val):
		self.value = val
		self.grad = 0


class Network:
	def __init__(self):
		print 'initializing network...'
		self.inputLayer = [InputUnit(0.3), InputUnit(0.3)]
		
		self.connectedLayers = []
		
		self.addConnectedLayer(3)
		self.addConnectedLayer(3)

		self.output = Neuron(self.connectedLayers[-1])
		
		self.mse = MSEUnit(self.output, 0)
		print 'network ready'

	def addConnectedLayer(self, num_neurons):
		if len(self.connectedLayers) > 0:
			lastLayer = self.connectedLayers[-1]
		else:
			lastLayer = self.inputLayer;

		newLayer = []

		for i in range(0, num_neurons):
			newLayer.append(Neuron(lastLayer))

		self.connectedLayers.append(newLayer)

	def forward(self, input_arr):
		inputs = []
		for index in range(0,len(input_arr)):
			self.inputLayer[index].value = input_arr[index]

		for layer in self.connectedLayers:
			for neuron in layer:
				neuron.forward()

		self.output.forward()

		return self.output.value

	def backward(self):
		self.mse.backward()
		self.output.backward()


		for layer in reversed(self.connectedLayers):
			for neuron in layer:
				neuron.backward()

	def updateParams(self):
		self.output.updateParams()

		for layer in reversed(self.connectedLayers):
			for neuron in layer:
				neuron.updateParams()


	def train(self, input_arr, label):
		self.forward(input_arr)
		
		self.mse.targetValue = label
		self.mse.forward()

		self.backward()
		self.updateParams()

		return self.mse.value
	


network = Network()

print network.forward([0,1])
epoch = 0
error = 5

while error > 0.0001:
	epoch += 1

	if epoch > 25000:
		LEARNING_RATE = 0.01
	error = 0
	error += network.train([0,0],0)
	error += network.train([1,0],0)
	error += network.train([0,1],0)
	error += network.train([1,1],1)

	if epoch % 100 == 0:
		print 'epoch : ', epoch, '\t error: ', error


print network.forward([0,0])
print network.forward([1,0])
print network.forward([0,1])
print network.forward([1,1])






