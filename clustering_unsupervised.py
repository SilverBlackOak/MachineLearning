from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

scaler = StandardScaler()
kmeans = KMeans(n_clusters=4)

pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)

labels = pipeline.predict(samples)
df = pd.DataFrame({'labels': labels, 'species':species})
ct = pd.crosstab(df['labels'], df['species'])

print(ct)
