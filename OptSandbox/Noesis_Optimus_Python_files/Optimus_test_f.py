a = 1
def f(x):
    return x ** 2
b = f(a)

open('output.txt', 'w').write(str(b))