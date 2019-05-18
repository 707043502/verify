
from helper import get_test, get_train, plot_decision_boundary
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier


if __name__== "__main__":
    sparse_data, sparse_label = get_train()
    dense_data = get_test()

    plot_decision_boundary(sparse_data, sparse_label, title="original data y=x+2")
    clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,
                                     max_depth=1, random_state=0).fit(sparse_data, sparse_label)
    dense_label = clf.predict(dense_data)

    plot_decision_boundary(dense_data, dense_label, title="original generalization")

    sparse_data, sparse_label = get_train()
    sparse_data['y'] = sparse_data['y'] - sparse_data['x']
    plot_decision_boundary(sparse_data, sparse_label, title="transformed data [x, y-x]")

    clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,
                                     max_depth=1, random_state=0).fit(sparse_data, sparse_label)

    dense_data = get_test()
    dense_data['y'] = dense_data['y'] - dense_data['x']
    dense_label = clf.predict(dense_data)
    plot_decision_boundary(dense_data, dense_label, title="transformed generalization")

    recover = dense_data
    recover['y'] = recover['y'] + recover['x']
    plot_decision_boundary(recover, dense_label, title="recovered generalization")

    plt.show()