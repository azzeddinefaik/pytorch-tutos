import torch

# When training neural networks, the most frequently used algorithm is back propagation. 
# In this algorithm, parameters (model weights) are adjusted according to the gradient of the loss function with respect to the given parameter.

# Consider the simplest one-layer neural network, with input x, parameters w and b, and some loss function.
# It can be defined in PyTorch in the following manner:
# Y = W*X + B

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# In this network, w and b are parameters, which we need to optimize. 
# Thus, we need to be able to compute the gradients of loss function with respect to those variables. 
# In order to do that, we set the requires_grad property of those tensors.

# A function that we apply to tensors to construct computational graph is in fact an object of class Function. 
# This object knows how to compute the function in the forward direction, 
# and also how to compute its derivative during the backward propagation step. 
# A reference to the backward propagation function is stored in grad_fn property of a tensor. 

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# optimize weights of parameters in the neural network, 
# we need to compute the derivatives of our loss function with respect to parameters, namely, we need &loss/&w and &loss/&b under some fixed values of x and y. To compute those derivatives, we call loss.backward(), and then retrieve the values from w.grad and b.grad:

loss.backward()
print(w.grad)
print(b.grad)
