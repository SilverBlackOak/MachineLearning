from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.4, normalize=True)

lasso.fit(X,y).coef_

lasso_coef = lasso.fit(X,y).coef_
print(lasso_coef)

plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()
