from network import Network

# Setting up Network
network = Network(learning_rate = 0.07)
network.addInputLayer(2)		
network.addConnectedLayer(5)
network.addConnectedLayer(5)
network.addConnectedLayer(1)
network.addMSE()


print network.forward([0,1])

epoch = 0
error = 5

while error > 0.001:
	epoch += 1

	error = 0
	error += network.train([0.1, 0.1], [0.2])
	error += network.train([0.2, 0.3], [0.5])
	error += network.train([0.6, 0.1], [0.7])
	error += network.train([0.4, 0.5], [0.9])

	if epoch % 100 == 0:
		print 'epoch : ', epoch, '\t error: ', error


print network.forward([0.1, 0.4])