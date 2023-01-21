##################################
#Randomized/Monte Carlo - MCCV Cross Validation - Rastgele Capraz Dogrulama
from sklearn.model_selection import ShuffleSplit

X = [...] # Veri kumesi verileri. / Ornegin; Iris=Setosa=0.42 ... vb.
y = [...] # Veri kumesi etiketleri. / Ornegin; Iris=Setosa,Versicolor... vb.

rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

for train_index, test_index in rs.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = ...  # Model
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(accuracy)

##################################
#K-Fold Cross Validation - K Katli/Katmanli Capraz Dogrulama
from sklearn.model_selection import KFold

X = [...] # Veri kumesi verileri. / Ornegin; Iris=Setosa=0.42 ... vb.
y = [...] # Veri kumesi etiketleri. / Ornegin; Iris=Setosa,Versicolor... vb.

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = ...  # Model
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(accuracy)

##################################
#Leave One/P Out Cross Validation - Biri (P) Disinda/Disarida Bırakilan Capraz Dogrulama
from sklearn.model_selection import LeaveOneOut

X = [...] # Veri kumesi verileri. / Ornegin; Iris=Setosa=0.42 ... vb.
y = [...] # Veri kumesi etiketleri. / Ornegin; Iris=Setosa,Versicolor... vb.

loo = LeaveOneOut()

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = ...  # Model
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(accuracy)

##################################
#Simple Cross Validation - Basit Capraz Dogrulama
from sklearn.model_selection import train_test_split

X = [...] # Veri kumesi verileri. / Ornegin; Iris=Setosa=0.42 ... vb.
y = [...] # Veri kumesi etiketleri. / Ornegin; Iris=Setosa,Versicolor... vb.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = ...  # Model
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(accuracy)

##################################
#Repeated K-Fold Cross Validation - Tekrarli K Katmanli Capraz Dogrulama
from sklearn.model_selection import RepeatedKFold

X = [...] # Veri kumesi verileri. / Ornegin; Iris=Setosa=0.42 ... vb.
y = [...] # Veri kumesi etiketleri. / Ornegin; Iris=Setosa,Versicolor... vb.

rkf = RepeatedKFold(n_splits=5, n_repeats=10)

for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = ...  # Model
    model.fit(X_train, y_train)    
    accuracy = model.score(X_test, y_test)
    print(accuracy)

##################################
#Holdout Cross Validation - Ayirarak Capraz Dogrulama
from sklearn.model_selection import train_test_split

X = [...] # Veri kumesi verileri. / Ornegin; Iris=Setosa=0.42 ... vb.
y = [...] # Veri kumesi etiketleri. / Ornegin; Iris=Setosa,Versicolor... vb.

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2 ,random_state=0)

model = ...  # Model
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(accuracy)

##################################
#Rolling/Rolling Window Cross Validation - Kaydirmali/Kaydirmali Bakis ile Capraz Dogrulama
from sklearn.model_selection import TimeSeriesSplit

X = [...] # Veri kumesi verileri. / Ornegin; Iris=Setosa=0.42 ... vb.
y = [...] # Veri kumesi etiketleri. / Ornegin; Iris=Setosa,Versicolor... vb.

tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = ...  # Model
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(accuracy)

##################################
#Bootstrap Cross Validation - Onyuklemeli Capraz Dogrulama
from sklearn.utils import resample

X = [...] # Veri kumesi verileri. / Ornegin; Iris=Setosa=0.42 ... vb.
y = [...] # Veri kumesi etiketleri. / Ornegin; Iris=Setosa,Versicolor... vb.

for i in range(10):
    X_resample, y_resample = resample(X, y)
    model = ...  # Model
    model.fit(X_resample, y_resample)
    accuracy = model.score(X, y)
    print(accuracy)