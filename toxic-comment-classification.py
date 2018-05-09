#Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

#Importing the Data
datatrain = pd.read_csv('train.csv')
datatrain = datatrain.replace(',','')

#Data PreProcessing

corpus = [] 
for i in range(0,20000):
    review = re.sub(r'[^a-zA-Z\s]', '', str(datatrain['comment_text'][i]))
    review = review.lower()
    review = review.strip()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ''.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
x= cv.fit_transform(corpus).toarray()

#Executing the above codes we get about 31000 features. But we want only the most used features and eliminate the rarely used features.

#Reducing the Features to 15000 and Creating the Bag of Words Model
cv=CountVectorizer(max_features=15000)
x= cv.fit_transform(corpus).toarray()
y= datatrain.iloc[0:20000,2].values

# Creating Topic Modelling
from sklearn.decomposition import LatentDirichletAllocation as LDA
lda = LDA(n_topics = 4, max_iter=100, random_state=100)
x_ld = lda.fit_transform(x)
features = pd.DataFrame(x_ld, columns = ['T1', 'T2', 'T3', 'T4'])


vocab = cv.get_feature_names()
tt_matrix = lda.components_
for topic_weights in tt_matrix:
    topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]
    topic = sorted(topic, key=lambda x: -x[1])
    topic = [item for item in topic if item[1] > 0.6]
    print(topic)

from sklearn.cluster import KMeans

#K-Means Clustering
km = KMeans(n_clusters=4, random_state=0)
km.fit_transform(features)
cluster_labels = km.labels_
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel']) #Y-value
kmeansdata = pd.concat([datatrain, cluster_labels], axis=1)

km.fit(x_ld)
y_kmeans = km.predict(x_ld)
plt.scatter(x_ld[:, 0], x_ld[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

#Visualizing the Topics
from numpy.random import randn
from scipy import stats
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

#HeatMap of the Topics
fig, ax = plt.subplots(figsize=(10, 10))        
sns.heatmap(features, ax=ax)
plt.show()

#Splitting the Data

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
import itertools
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score


classifier = GaussianNB()
classifier.fit(x_train, y_train)

#Partial Fit for Naive Bayes
n = 20000
nbatch1 = 100

#Predicting the test set results for Naive Bayes
from sklearn.metrics import confusion_matrix

for i in range(int(n/nbatch1)):
    classifier.partial_fit(x[i*nbatch1:(i*nbatch1+nbatch1)],y[i*nbatch1:(i*nbatch1+nbatch1)], classes = np.unique(y))
    y_pred_naive = classifier.predict(x_test)

    cm_naive = confusion_matrix(y_test, y_pred_naive)
print(cm_naive)
def plot_confusion_matrix(cm_naive, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm_naive = cm_naive.astype('float') / cm_naive.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm_naive)

    plt.imshow(cm_naive, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_naive.max() / 2.
    for i, j in itertools.product(range(cm_naive.shape[0]), range(cm_naive.shape[1])):
        plt.text(j, i, format(cm_naive[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_naive[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
plot_confusion_matrix(cm_naive, [0,1],normalize=True)
def ROC(label,result):
    # Compute ROC curve and area the curve
    Y = np.array(label)
    fpr, tpr, thresholds = roc_curve(Y, result)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    # Plot ROC curve
    #pl.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
y_score_naive = classifier.predict_proba(x_test)[:,1]
ROC(y_test,y_score_naive)

print(classification_report(y_test,y_pred_naive))

print(matthews_corrcoef(y_test,y_pred_naive))

classifier.score(x_test, y_test)

#Learning Curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

plot_learning_curve(classifier, title, x_train,y_train,ylim=(0.7, 1.01))
plt.show()

#SGD Classifier

#Fitting SGD to the training set
from sklearn.linear_model import SGDClassifier

#Implementing Grid Search to Find the Best Parameters of SGD

param_test1_sgd = {'loss':["log", "hinge"], 'penalty':["l2", "l1", "elasticnet"], 'alpha':[0.00001,0.0001, 0.001, 0.1]}

from sklearn.model_selection import GridSearchCV

gsearch_sgd = GridSearchCV(estimator = SGDClassifier(learning_rate='optimal', random_state=0),
                           param_grid= param_test1_sgd,
                           scoring='accuracy',n_jobs=4, cv=5)

gsearch_sgd.fit(x_train, y_train)
score = gsearch_sgd.score(x_test, y_test)    
print('Accuracy:{}, Best Parameters:{}'.format(score,gsearch_sgd.best_params_))
print(gsearch_sgd.best_params_)

#Fitting the SGD Classifier to the Dataset using the Best Parameters
sgd = SGDClassifier(learning_rate='optimal',alpha=0.001, loss = 'hinge', penalty = 'l2')
sgd.fit(x_train, y_train)
from sklearn.metrics import f1_score
train_score=[]
test_score=[]
iterations=[]

#Partial Fit
n = 20000
nbatch1 = 100

from sklearn.metrics import confusion_matrix

#Predicting the test set results
for i in range(int(n/nbatch1)):
    sgd.partial_fit(x[i*nbatch1:(i*nbatch1+nbatch1)],y[i*nbatch1:(i*nbatch1+nbatch1)], classes = np.unique(y))
    y_pred_sgd = sgd.predict(x_test)
    train_auc=f1_score(y_train, sgd.predict(x_train))
    test_auc=f1_score(y_test,sgd.predict(x_test))
    train_score.append(train_auc)
    test_score.append(test_auc)

    cm_sgd = confusion_matrix(y_test, y_pred_sgd)
print(cm_sgd)
plot_confusion_matrix(cm_sgd, [0,1],normalize=True)
plt.plot(train_score)
plt.plot(test_score)
plt.legend(('train','test'))
plt.show()

print(classification_report(y_test,y_pred_sgd))

print(matthews_corrcoef(y_test,y_pred_sgd))
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
title = "Learning Curve(SVM)"
plot_learning_curve(sgd, title, x_train,y_train,ylim=(0.7, 1.01))
plt.show()

sgd.score(x_test, y_test)

#Random Forest

from sklearn.ensemble import RandomForestClassifier

#Implementing Grid Search to Find the Best Parameters of Random Forest

param_grid_forest = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
random_forest =  RandomForestClassifier(n_estimators=10,random_state=42)

gsearch_forest =  RandomizedSearchCV(estimator = random_forest,
                                                param_distributions = param_grid_forest,
                                                n_iter = 100, cv = 3, verbose=2, 
                                                random_state=42, n_jobs = -1)

#Fitting the Random Forest to the Dataset
gsearch_forest= gsearch_forest.fit(x_train, y_train)
score = gsearch_forest.score(x_test, y_test)    
print('Accuracy:{}, Best Parameters:{}'.format(score,gsearch_forest.best_params_))
print(gsearch_forest.best_params_)

#Confusion Matrix for Random Forest

from sklearn.metrics import confusion_matrix

y_pred_random = gsearch_forest.predict(x_test)
cm_random = confusion_matrix(y_test, y_pred_random)

gsearch_forest.score(x_test, y_test)

print(cm_random)
plot_confusion_matrix(cm_random, [0,1],normalize=True)
y_score_random = gsearch_forest.predict_proba(x_test)[:,1]
ROC(y_test,y_score_random)
print(classification_report(y_test,y_pred_random))

print(matthews_corrcoef(y_test,y_pred_random))
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
title = "Learning Curve Random Forest"
plot_learning_curve(gsearch_forest, title, x_train,y_train,ylim=(0.7, 1.01))
plt.show()

#Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
import itertools
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score

#Implementing Grid Search to Find the Best Parameters of SGD

param_test1_mnb = {'alpha':[0.0001, 0.001, 0.1, 1], 
                   "fit_prior": [True, False]}

from sklearn.model_selection import GridSearchCV

gsearch_mnb = GridSearchCV(estimator = MultinomialNB(),
                           param_grid= param_test1_mnb,
                           scoring='accuracy',n_jobs=4, cv=5)

gsearch_mnb.fit(x_train, y_train)
score = gsearch_mnb.score(x_test, y_test)    
print('Accuracy:{}, Best Parameters:{}'.format(score,gsearch_mnb.best_params_))
print(gsearch_mnb.best_params_)

#Using the Best Parameters inside the Classifier
classifiermnb = MultinomialNB(alpha =1, fit_prior=True )
#Partial Fit for Naive Bayes
n = 20000
nbatch1 = 100

for i in range(int(n/nbatch1)):
    classifiermnb.partial_fit(x[i*nbatch1:(i*nbatch1+nbatch1)],y[i*nbatch1:(i*nbatch1+nbatch1)], classes = np.unique(y))
    y_pred_naivemnb = classifiermnb.predict(x_test)

    cm_naivemnb = confusion_matrix(y_test, y_pred_naive)
print(cm_naivemnb)
plot_confusion_matrix(cm_naivemnb, [0,1],normalize=True)

    
y_score_naivemnb = classifiermnb.predict_proba(x_test)[:,1]
ROC(y_test,y_score_naivemnb)

print(classification_report(y_test,y_pred_naivemnb))

print(matthews_corrcoef(y_test,y_pred_naivemnb))

title = "Learning Curves (Naive Bayes_Multinomial)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

plot_learning_curve(classifiermnb, title, x_train,y_train,ylim=(0.7, 1.01))
plt.show()

classifiermnb.score(x_test, y_test)

#Multi Layer Perception (MLP)
from sklearn.neural_network import MLPClassifier
classifiermlp = MLPClassifier(solver='adam',hidden_layer_sizes=30,alpha=1e-04)
for i in range(int(n/nbatch1)):
    classifiermlp.partial_fit(x[i*nbatch1:(i*nbatch1+nbatch1)],y[i*nbatch1:(i*nbatch1+nbatch1)], classes = np.unique(y))
    y_pred_mlp = classifiermlp.predict(x_test)

    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
print(cm_mlp)
y_score_mlp = classifiermlp.predict_proba(x_test)[:,1]
ROC(y_test,y_score_mlp)
plt.figure()
print(classification_report(y_test,y_pred_mlp))
print(matthews_corrcoef(y_test,y_pred_mlp))

plt.figure()
plot_confusion_matrix(cm_mlp, [0,1],normalize=True)
plt.show()

classifiermlp.score(x_test, y_test)

plt.figure()
title = "Learning Curves (Multi-Layer Perception)"
plot_learning_curve(classifiermlp, title, x_train,y_train,ylim=(0.7, 1.01))
plt.show()