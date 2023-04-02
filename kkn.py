import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data = pd.read_csv('iris.csv')
df = pd.DataFrame(data= data)
# print(data)
features = df.iloc[:, :4]
labels = df.iloc[:, 4]
# print(labels)
# print(features)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

net = KNeighborsClassifier(n_neighbors=7)
H = net.fit(X_train, y_train)

y_pred = net.predict(X_test)
# print(y_pred)
# print(y_test)
acc = accuracy_score(y_pred, y_test)
print(f'Accuracy: {acc * 100 :.2f} %')

