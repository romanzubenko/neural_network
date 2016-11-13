import math

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