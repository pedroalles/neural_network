from sklearn.neural_network import MLPClassifier
import numpy as np

# X = [[0., 0.], [1., 1.]]

# y = [0, 1]

X = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
            
y = [0, 1, 1, 0]

clf = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(100,2),
                    activation='logistic',
                    random_state=1)

clf.fit(X, y)

predict = clf.predict([[0, 1, 1], [1, 1, 1], [1, 0, 0]])
predict_log = clf.predict_log_proba([[0, 1, 1], [1, 1, 1], [1, 0, 0]])

score = clf.score(X, y)

print(score)
print(predict)
print(predict_log)

clf.coefs_

[print(coef.shape) for coef in clf.coefs_]

