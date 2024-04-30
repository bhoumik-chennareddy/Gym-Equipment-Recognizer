import pandas as pd
train_labels = pd.read_csv('C:/Users/bhoum/OneDrive/Desktop/Machine Learning/1 Gym Equipment.v1i.retinanet/train/_annotations.csv', header=None)
valid_labels = pd.read_csv('C:/Users/bhoum/OneDrive/Desktop/Machine Learning/1 Gym Equipment.v1i.retinanet/valid/_annotations.csv', header=None)

train_labels[5] = train_labels[5].replace('json', 'null')
valid_labels[5] = valid_labels[5].replace('json', 'null')

train_labels.to_csv('cleaned_train_annotations.csv', index=False, header=False)
valid_labels.to_csv('cleaned_valid_annotations.csv', index=False, header=False)

train_labels = pd.read_csv('C:/Users/bhoum/OneDrive/Desktop/Machine Learning/1 Gym Equipment.v1i.retinanet/train/_annotations.csv', header=None)
valid_labels = pd.read_csv('C:/Users/bhoum/OneDrive/Desktop/Machine Learning/1 Gym Equipment.v1i.retinanet/valid/_annotations.csv', header=None)


print("Before replacement:")
print(train_labels[5].unique())
print(valid_labels[5].unique())


train_labels[5] = train_labels[5].replace('json', 'null')
valid_labels[5] = valid_labels[5].replace('json', 'null')


print("\nAfter replacement:")
print(train_labels[5].unique())
print(valid_labels[5].unique())


train_labels.to_csv('cleaned_train_annotations.csv', index=False, header=False)
valid_labels.to_csv('cleaned_valid_annotations.csv', index=False, header=False)

print("\nCSV files cleaned and saved successfully!")
