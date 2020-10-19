import numpy as np

def nonlin(x):
    return 1 / (1 + np.exp(-x))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([[0], [1], [1], [1]])

np.random.seed(1)

syn0 = 2*np.random.random((2, 2)) - 1
syn1 = 2*np.random.random((2, 1)) - 1

print('Initial weights')
print(syn0)

# training step
for j in range(60000):

    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    l2_error = y - l2
    l2_delta = l2_error * nonlin(l2)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

    # print('Step ' + str(j + 1))
    # print(syn0)

    if (j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(l2_error))))

print("This is the output when the training is finished")
print(l2)