###########################
#PREPARE THE DATASET
#    The slide_id is an arbitrary identifier for each slide.
#    The class column refers to the WSI's class in the classification task
#    The label column is the WSI's class but coded as an integer label for use under the hood in Pytorch.
#    The fold-0 ... fold-k columns tell train.py whether the WSI should be part of the training or validation set during each fold of the cross validation routine. The number of fold-* columns should correspond to the number of folds you want to use in the k-fold cross validation routine. E.g. when you want k = 6 the columns fold-0 through fold-5 should be present.
###########################

#%%
import pandas as pd

df = pd.read_csv('/mnt/bulk-curie/arianna/HECTOR/HECTOR/preprocess_table.csv')

print(df.head(10))

df = df.rename(columns={'stage': 'class'})
df['label'] = df['class'].map({'high': 1, 'low': 0})

print(df.head(10))

# %%
import numpy as np

# Calculate the number of rows
n_rows = len(df)

# Calculate the number of training samples (80% of total)
n_train = int(0.8 * n_rows)

# Create new columns for each fold
for i in range(5):
    # Shuffle the indices for this fold
    idx = np.random.permutation(n_rows)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    # Assign labels to the new column
    df[f'fold-{i}'] = 'validation'
    df.loc[train_idx, f'fold-{i}'] = 'training'

print(df.head(10))

#%%
#save the dataframe in the right format
df = df[['slide_id', 'class', 'label', 'fold-0', 'fold-1', 'fold-2', 'fold-3', 'fold-4']]
 
print(df.head(10))

#save the dataframe as train.csv
df.to_csv('/mnt/bulk-curie/arianna/HECTOR/HECTOR/train.csv', index=False)


