import pandas as pd
import numpy as np

digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',
                           header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',
                          header=None)

# x_digits = digits_train[np.arange(64)]
# y_digits = digits_train[64]
#
from sklearn.decomposition import PCA
#
# estimator = PCA(n_components=2)
# x_pca = estimator.fit(x_digits)
#
# import matplotlib.pyplot as plt
#
#
# def plot_pca_scatter():
#     colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
#     for i in np.arange(len(colors)):
#         px = x_pca[:, 0][y_digits.as_matrix() == i]
#         py = x_pca[:, 1][y_digits.as_matrix() == i]
#         plt.scatter(px, py, c=colors[i])
#     plt.legend(np.arange(0, 10).astype(str))
#     plt.xlabel('First Principal Component')
#     plt.ylabel('Second Principal Component')
#     plt.show()
#
#
# plot_pca_scatter()

x_train=digits_train[np.arange(64)]
y_train=digits_train[64]
x_test=digits_test[np.arange(64)]
y_test=digits_test[64]

from sklearn.svm import LinearSVC
svc=LinearSVC()
svc.fit(x_train,y_train)
y_predict=svc.predict(x_test)

estimator=PCA(n_components=20)#64维转换为20维
pca_x_train=estimator.fit_transform(x_train)
pca_x_test=estimator.transform(x_test)
pca_svc=LinearSVC()
pca_svc.fit(pca_x_train,y_train)
y_predict=pca_svc.predict(pca_x_test)

from sklearn.metrics import classification_report
print(svc.score(x_test,y_test))
print(pca_svc.score(pca_x_test,y_test))
print(classification_report(y_test,y_predict,target_names=np.arange(10).astype(str)))
