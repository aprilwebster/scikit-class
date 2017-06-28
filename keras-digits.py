from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train is a big array of pixels
# digit is the digit it is - 0..9
# image is 128 by 128: 0,0 in top LHcorner to 0,128 in top RHcorner

digit = X_train[0]
print(digit.shape)
str = ""
for i in range(digit.shape[0]):
    for j in range(digit.shape[1]):
        if digit[i][j] == 0:
            str += " "
        elif digit[i][j] < 128:
            str += "."
        else:
            str += "X"
    str += "\n"

print(str)
print("Label: ", y_train[0])
