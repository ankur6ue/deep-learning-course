import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')

# This program takes some MNIST digits data, reshapes the 28*28 image array into a 784*1 dimension vector and
# maps it to 2D using the t-sne algorithm. Then it plots the resulting 2D point cloud. The points corresponding to
# distinct digits are clustered together

# data from sklearn datasets
data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

# Extract data & target from the dataset
pixel_data, targets = data
targets = targets.astype(int)

# Read one row, convery to numpy and reshape
single_image = pixel_data.iloc[5, :].values.reshape(28,28)

plt.imshow(single_image, cmap='gray')
plt.title(f"Image of the text: {targets[5]}", fontsize=15)
plt.show()

# Object of tSNE
tsne = TSNE(n_components=2, random_state=42)

x_transformed = tsne.fit_transform(pixel_data[:3000, :].values) # Data upto 3000 rows

# convert the transformed data into dataframe
tsne_df = pd.DataFrame(np.column_stack((x_transformed, targets[:3000])), columns=['X', 'Y', "Targets"])

tsne_df.loc[:, "Targets"] = tsne_df.Targets.astype(int)

plt.figure(figsize=(10,8))

g = sns.FacetGrid(data=tsne_df, hue='Targets', height=8)

g.map(plt.scatter, 'X', 'Y').add_legend()

plt.show()