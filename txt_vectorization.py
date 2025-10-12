#!/usr/bin/env python
# coding: utf-8

# In[34]:


import nltk
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from nltk import tokenize
from scipy.sparse import hstack


# In[35]:


# Read the CSV file
data = pd.read_csv("training_data_clean.csv")

# Initialize CountVectorizer
vectorize_1 = data.iloc[:, 1]
vectorize_2 = data.iloc[:, 6]
vectorize_3 = data.iloc[:, 9]


# In[36]:


data


# In[37]:


vectorize_1


# In[38]:


vectorize_2


# In[39]:


vectorize_3


# In[40]:


# each life fits the countervectorier on the text column and transforms it into a numeric
#matrix(used seperate countvectorizers and merge vocabs)
# so each row is one survey response and each column is one unique word in our dataset
cv1 = CountVectorizer()
cv2 = CountVectorizer()
cv3 = CountVectorizer()

X1 = cv1.fit_transform(vectorize_1.astype(str))
X2 = cv2.fit_transform(vectorize_2.astype(str))
X3 = cv3.fit_transform(vectorize_3.astype(str))


# In[41]:


# combined all three text matrices side by side so that each text feature from each column
# becomes part of one big feature matrix
X_combined = hstack([X1, X2, X3])
# 825 data samples(approx 3*N) where N = number of student responses
# 6105 unique workds accross all three columns(X1,X2,X3)
print("Shape of combined matrix:", X_combined.shape)


# In[42]:


X_combined
# Here 51009 stored element => number of non-zero word counts accross all documents


# In[43]:


X1


# In[44]:


cv.get_feature_names_out() 
# each string here corresponds to one column in our vectorized matrix


# In[45]:


cv.vocabulary_['able'] # outputs the index


# In[46]:


cv.get_feature_names_out()[:50] # this would show first 50 feature names


# In[48]:


feature_names = list(cv1.get_feature_names_out()) + \
                list(cv2.get_feature_names_out()) + \
                list(cv3.get_feature_names_out())
X_dense = X_combined.toarray()

#Turn it into a DataFrame for readability
import pandas as pd
df_matrix = pd.DataFrame(X_dense, columns= feature_names)

#View the first few rows
print(df_matrix.head(5))


# In[ ]:




