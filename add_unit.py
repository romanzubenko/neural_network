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