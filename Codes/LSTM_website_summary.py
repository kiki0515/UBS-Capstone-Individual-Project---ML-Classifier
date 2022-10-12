import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import tensorflow as tf
import itertools
import tensorflow_addons as tfa
from sklearn.metrics import f1_score

#path1 = 'E:/nlp/data2.parquet'
#path2 = 'E:/nlp/data3.parquet'
path3 = 'E:/nlp/data4.parquet'


df3 = pd.read_parquet(path3)
print(df3.head())
print(df3.columns)

## naive baseline
df3['target'].value_counts()
naive_predict = np.zeros(df3.shape[0])
assert len(df3.target.values) == len(naive_predict)
print(f1_score(df3.target.values,naive_predict,average='weighted'))


### Plot hist for length of column
plt.hist(df3['website_summary'].apply(len),bins=500)  # here density=False would make counts, density=True make probability
plt.ylabel('Frequency')
plt.xlabel('Data')
plt.show()

### Plot CDF PDF for length of column
count, bins_count = np.histogram(df3['website_summary'].apply(len).values, bins=500)
pdf = count / sum(count)  # finding the PDF of the histogram using count values
cdf = np.cumsum(pdf) # using numpy np.cumsum to calculate the CDF, We can also find using the PDF values by looping and adding
# plotting PDF and CDF
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.legend()

### quantile of length for all observation
print(np.quantile(df3['website_summary'].apply(len).values, 1))  # 33 for 0.95
print(np.quantile(df3['website_summary'].apply(len).values, .5))  # 15 for 0.5

# LSTM Model
label = df3['target'].copy()
print(label.head())
print(label.value_counts()) #12443:5038
print(df3.isna().sum())  # no nan in features
print(label.isna().sum())  # no nan in labels
assert (df3.shape[0]) == (label.shape[0])
print('percent of 0 in y_train:', label.value_counts()[0]/label.shape[0]) # model will be fine if accuracy > 0.712



X = [l.tolist() for l in df3.website_summary.values]
tokenizer = Tokenizer() #the maximum number of words to be used (most frequent)
tokenizer.fit_on_texts(X)
vocab_size = len(tokenizer.word_index) + 1 #vocabulary size
encoded_docs = tokenizer.texts_to_sequences(X)
padded_sequence_X = pad_sequences(encoded_docs, maxlen=50)  # same length for observations
print('vocabulary size', vocab_size)
print('vocabulary:\n', dict(itertools.islice(tokenizer.word_index.items(), 15)))
print(encoded_docs[0])

Y = pd.get_dummies(label.values)
Y = np.array(Y)
print(Y, Y.shape)

# returns training and test subsets that have the same proportions of class label
X_train,X_test,y_train,y_test = train_test_split(padded_sequence_X, Y, test_size=0.3, stratify=Y)
#print('#0/#1 in y_train:', y_train.value_counts()[0]/y_train.value_counts()[1])
#print('#0/#1 in y_test:', y_test.value_counts()[0]/y_test.value_counts()[1])
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
# One-Hot Encoding for y_train
#a = y_train_0.values
#y_train = np.zeros((a.size, a.max()+1))
#y_train[np.arange(a.size),a] = 1
#print(y_train,y_train.shape)


# Build the model
embedding_vector_length = 64 # dimension for each word
model = Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_vector_length, input_length=50))
model.add(tf.keras.layers.SpatialDropout1D(0.25))
model.add(tf.keras.layers.LSTM(256, dropout=0.5, recurrent_dropout=0.5))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(6, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
#es = EarlyStopping(monitor='val_loss', patience=3)
f1 = tfa.metrics.F1Score(num_classes=2, average='weighted',threshold=0.5)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=6, batch_size=32)

