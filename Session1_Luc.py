# exercise 1
import numpy as np
from tensorflow.keras import backend as K

a = K.placeholder(shape=(5,))
b = K.placeholder(shape=(5,))
c = K.placeholder(shape=(5,))

input_squared_a = a ** 2
input_squared_b = b ** 2
input_squared_c = c ** 2
input_mul = 2 * b * c
input_added = input_squared_a + input_squared_b + input_squared_c + input_mul

elem_wise_function = K.function(inputs=[a,b,c],
                          outputs=input_added)

print(elem_wise_function((np.arange(5).reshape((5,1)),np.arange(5).reshape((5,1)),np.arange(5).reshape((5,1)))))

#exercise 2
x = K.placeholder(shape=())
tanh = (K.exp(2*x) - 1)/(K.exp(2*x) + 1)
tanh_func = K.function(inputs=[x], outputs=tanh)

print(tanh_func((-100)))
print(tanh_func((-1)))
print(tanh_func((0)))
print(tanh_func((1)))
print(tanh_func((100)))

grad_1_tensor = K.gradients(loss=tanh, variables=[x])
grad_1_functions = K.function(inputs=[x], outputs=grad_1_tensor[0])

print(grad_1_functions((-100)))
print(grad_1_functions((-1)))
print(grad_1_functions((0)))
print(grad_1_functions((1)))
print(grad_1_functions((100)))

#exercise 3
w = K.placeholder(shape=(2,))
b = K.placeholder(shape=(1,))
x1 = K.placeholder(shape=(2,))
arg = w[0] * x1[0] + w[1] * x1[1] + b[0]

f = 1 / (1 + K.exp(-arg))
f_func = K.function(inputs=[w,b,x1], outputs=f)

grad_f_tensor = K.gradients(loss=f, variables = [w])
grad_f_functions = K.function(inputs=[w,b,x1], outputs=grad_f_tensor[0])

print(grad_f_functions((np.arange(2,4),np.arange(3,4),np.arange(5,7))))

#exercise 4
x2 = K.placeholder(shape=())
