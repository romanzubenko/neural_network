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