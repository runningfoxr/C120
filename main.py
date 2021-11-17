from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

data = datasets.load_wine()
x = data.data
y = data.target

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.25, random_state=42)

stan_scal = StandardScaler()

x_train = stan_scal.fit_transform(x_train)
x_test = stan_scal.fit_transform(x_test)


gauss_nb = GaussianNB()
gauss_nb.fit(x_train, y_train)

y_pred = gauss_nb.predict(x_test)

print("The accuracy is: ", accuracy_score(y_test, y_pred))


