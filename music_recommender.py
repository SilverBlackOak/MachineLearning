from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline
import pandas as pd

scaler = MaxAbsScaler()
nmf = NMF(n_components=20)
normalizer = Normalizer()

pipeline = make_pipeline(scaler, nmf, normalizer)

norm_features = pipeline.fit_transform(artists)

df = pd.DataFrame(norm_features, index=artist_names)
artist = df.loc['Bruce Springsteen']
similarities = df.dot(artist)

print(similarities.nlargest())
