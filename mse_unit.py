import math

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