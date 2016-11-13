import math
from neuron import Neuron
from mse_unit import MSEUnit
from input_unit import InputUnit


class Network:
	def __init__(self, learning_rate = 0.01):
		self.connectedLayers = []
		self.learning_rate = learning_rate

	def addInputLayer(self, inputSize):
		self.inputLayer = []
		for i in range(0, inputSize):
			self.inputLayer.append(InputUnit(0))

	def addMSE(self):
		self.mseLayer = []

		output = self.connectedLayers[-1]
		for neuron in output:
			self.mseLayer.append(MSEUnit(neuron, 0))


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
		
		for index in range(0, len(input_arr)):
			self.inputLayer[index].value = input_arr[index]

		for layer in self.connectedLayers:
			for neuron in layer:
				neuron.forward()

		output = self.connectedLayers[-1]
		return [neuron.value for neuron in output]

	def backward(self):
		for mse in self.mseLayer:
			mse.backward()

		for layer in reversed(self.connectedLayers):
			for neuron in layer:
				neuron.backward()

	def updateParams(self):
		for layer in reversed(self.connectedLayers):
			for neuron in layer:
				neuron.updateParams(self.learning_rate)


	def train(self, input_arr, labels):
		self.forward(input_arr)
		
		for index in range(0, len(labels)):
			self.mseLayer[index].targetValue = labels[index]
			self.mseLayer[index].forward()

		self.backward()
		self.updateParams()

		return self.mseLayer[0].value