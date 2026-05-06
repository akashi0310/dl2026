def f(x):
    return x**2

def df(x):
    return 2 * x

L = 0.01 # learning rate

def gradient_descent(x):
    x = x - L * df(x)
    return x

x = 10
print("Time   x      f(x)")
for i in range(1,11): 
    x = gradient_descent(x)
    if x == 1:
        break
    print(f"{i}      {x:.2f}   {f(x):.2f}")
    
