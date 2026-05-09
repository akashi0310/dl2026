import matplotlib.pyplot as plt
with open("C:/Users/lapla/OneDrive/Desktop/codespace/master/deeplearning/dl2026/lab2/lr.csv", "r", encoding="utf-8") as file:
    data = []
    for f in file:
        row = f.split(",") # 10, 55
        data.append([int(row[0]), int(row[1])]) 

print(data)

x_vals = [point[0] for point in data]
y_vals = [point[1] for point in data]

plt.scatter(x_vals, y_vals, color='blue', marker='o')
plt.title('Dataset Plot')
plt.xlabel('X')
plt.ylabel('Y')
# We will show the plot at the end


def loss(w1,w0,data):
    return 1/2 * 1/len(data)*sum((w1*data[i][0] + w0 - data[i][1])**2 for i in range(len(data)))

def df0(w1,w0,data):
    return sum(w1* data[i][0] + w0 - data[i][1] for i in range(len(data)))

def df1(w1,w0,data):
    return sum(data[i][0] * (w1* data[i][0] + w0 - data[i][1]) for i in range(len(data)))
    
def grad(w1,w0,data,lr,epochs):
    loss_value = []
    for epoch in range(epochs):
        w0 = w0 - lr * df0(w1,w0,data)
        w1 = w1 - lr * df1(w1,w0,data)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss(w1,w0,data):.4f}")
            plt.plot(x_vals, [w1*x + w0 for x in x_vals], color='gray', alpha=0.3)
            
        loss_value.append(loss(w1,w0,data))

        if (len(loss_value) > 1) and (abs(loss_value[-1] - loss_value[-2]) < 0.005):
            break
            
    return w1, w0

# Run gradient descent
w1, w0 = grad(w1=1, w0=0, data=data, lr=0.0001, epochs=5000)

print(f"Final Weights: w1 = {w1:.4f}, w0 = {w0:.4f}")

# Plot the final fitted line
plt.plot(x_vals, [w1*x + w0 for x in x_vals], color='red', label='Final Line')
plt.legend()
plt.show()
