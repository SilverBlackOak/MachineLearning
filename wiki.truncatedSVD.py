from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import pandas as pd

svd = TruncatedSVD(n_components=50)
kmeans = KMeans(n_clusters=6)

pipeline = make_pipeline(svd, kmeans)
pipeline.fit(articles)

labels = pipeline.predict(articles)

df = pd.DataFrame({'label': labels, 'article': titles})

print(df.sort_values('label'))
