import torch

x = torch.load("train.pt")

X = x["X"] # (X_i , Jij)
A = x["A"] # (h_i, Jij, theta)
y = x["y"] #(ZiZj)
print("inputs: (X_obs and Jij)")
print(X.shape)
print(X[0])

print("confirm hi, Jij, theta:")
print(A.shape)
print(A[0])

print("outputs")
print(y.shape)
print(y[0])
