import random

class Weight:
	def __init__(self):
		self.value = (random.random() * 2) - 1
		self.grad = 0

		print '\tinit weight..', self.value

	def forward(self):
		self.grad = 0
		self.value = self.value

	def updateParam(self, learning_rate):
		self.value += (-1) * self.grad * learning_rate