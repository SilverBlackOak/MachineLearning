from sklearn.linear_model import LinearRegression

lr = LogisticRegression()
lr.fit(X, y)

review1 = "LOVED IT! This movie was amazing. Top 10 this year."
review1_features = get_features(review1)
print("Review:", review1)
print("Probability of positive review:", lr.predict_proba(review1_features)[0,1])

review2 = "Total junk! I'll never watch a film by that director again, no matter how good the reviews."
review2_features = get_features(review2)
print("Review:", review2)
print("Probability of positive review:", lr.predict_proba(review2_features)[0,1])

review3 = "I guess the movie was amazing. There were some nice scenes though too predictable"
review3_features = get_features(review3)
print("Review:", review3)
print("Probability of positive review:", lr.predict_proba(review3_features)[0,1])
