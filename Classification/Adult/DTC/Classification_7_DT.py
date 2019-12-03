# Dataset 7: Adult
# AB: Decision Tree Classification
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.metrics import accuracy_score

data = np.loadtxt('adult.data', delimiter=',',dtype='str')
X_old=data[:,:14]
y_old=data[:,14:]
X_old=np.char.strip(X_old)
y_old=np.char.strip(y_old)

dictionary = {'?':'0','Private':'1','Self-emp-not-inc':'2','Self-emp-inc':'3','Federal-gov':'4',
              'Local-gov':'5','State-gov':'6','Without-pay':'7','Never-worked':'8',
              'Bachelors':'1','Some-college':'2','11th':'3','HS-grad':'4','Prof-school':'5',
              'Assoc-acdm':'6','Assoc-voc':'7','9th':'8','7th-8th':'9','12th':'10','Masters':'11',
              '1st-4th':'12','10th':'13','Doctorate':'14','5th-6th':'15','Preschool':'16',
              'Married-civ-spouse':'1','Divorced':'2','Never-married':'3','Separated':'4','Widowed':'5',
              'Married-spouse-absent':'6','Married-AF-spouse':'7',
              'Tech-support':'1','Craft-repair':'2','Other-service':'3','Sales':'4','Exec-managerial':'5',
              'Prof-specialty':'6','Handlers-cleaners':'7','Machine-op-inspct':'8','Adm-clerical':'9',
              'Farming-fishing':'10','Transport-moving':'11','Priv-house-serv':'12','Protective-serv':'13',
              'Armed-Forces':'14',
              'Wife':'1','Own-child':'2','Husband':'3','Not-in-family':'4','Other-relative':'5',
              'Unmarried':'6',
              'White':'1','Asian-Pac-Islander':'2','Amer-Indian-Eskimo':'3','Other':'4','Black':'5',
              'Female':'1','Male':'2',
              'United-States':'1','Cambodia':'2','England':'3','Puerto-Rico':'4','Canada':'5',
              'Germany':'6','Outlying-US(Guam-USVI-etc)':'7','India':'8','Japan':'9','Greece':'10',
              'South':'11','China':'12','Cuba':'13','Iran':'14','Honduras':'15','Philippines':'16',
              'Italy':'17','Poland':'18','Jamaica':'19','Vietnam':'20','Mexico':'21',
              'Portugal':'22','Ireland':'23','France':'24','Dominican-Republic':'25','Laos':'26',
              'Ecuador':'27','Taiwan':'28','Haiti':'29','Columbia':'30','Hungary':'31','Guatemala':'32',
              'Nicaragua':'33','Scotland':'34','Thailand':'35','Yugoslavia':'36','El-Salvador':'37',
              'Trinadad&Tobago':'38','Peru':'39','Hong':'40','Holand-Netherlands':'41'}
X=np.copy(X_old)
for old, new in dictionary.items():
    X[X_old==old] = new

dictionary_y = {'>50K':'1','<=50K':'0'}
y=np.copy(y_old)
for old, new in dictionary_y.items():
    y[y_old==old] = new

X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
scaler=sklearn.preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

parameters = {
    'max_depth': np.linspace(1,30,10, dtype=np.int32),
    'criterion': ["gini", "entropy"]
}
dtc=DecisionTreeClassifier(random_state=0)
randcv = RandomizedSearchCV(dtc, parameters, n_iter=50, verbose=1, random_state=0, cv=5)
randcv.fit(X_train, y_train)

print(randcv.best_estimator_)
print(randcv.best_score_)
print(randcv.best_params_)

y_test_pred = randcv.best_estimator_.predict(X_test)
print(accuracy_score(y_test, y_test_pred))