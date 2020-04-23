train_errs = list()
valid_errs = list()

for C_value in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    lr = LogisticRegression(C=C_value)
    lr.fit(X_train, y_train)
   
    train_errs.append( 1.0 - lr.score(X_train, y_train) )
    valid_errs.append( 1.0 - lr.score(X_valid, y_valid) )

plt.semilogx(C_values, train_errs, C_values, valid_errs)
plt.legend(("train", "validation"))
plt.show()
