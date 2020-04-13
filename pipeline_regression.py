from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

pipeline = Pipeline(steps)

parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

gm_cv = GridSearchCV(pipeline, parameters)
gm_cv.fit(X_train, y_train)

r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
