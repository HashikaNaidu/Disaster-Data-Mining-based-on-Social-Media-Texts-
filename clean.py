#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import string
import re
import nltk


# In[2]:


df = pd.read_csv("Pakistan floods data.csv", encoding = "utf-8")
df = df.drop(["Unnamed: 0"], axis=1)


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


dframe = pd.DataFrame(df[["tweets", "likes"]])


# In[6]:


# dframe.head()


# In[7]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, " ", input_txt)
        return input_txt
    
dframe["remove_user"] = np.vectorize(remove_pattern)(dframe["tweets"], "@[\w]*")
dframe


# In[8]:


def remove(txt):
    txt = re.sub("[0-9]+", " ", txt)
    txt = re.sub(r"\$\w*", " ", txt)
    txt = re.sub(r"^RT\s\w*", " ", txt)
    txt = re.sub(r"#", " ", txt)
    txt = re.sub(r"https ", " ", txt)
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', txt)
   
    return txt
dframe["remove_http"]  = dframe["remove_user"].apply(lambda x: remove(x))
dframe.sort_values("remove_http", inplace = True)
dframe.drop_duplicates(subset= "remove_http", keep = "first", inplace = True)


# In[23]:


dframe


# In[24]:


# import nltk
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))


# In[25]:


from nltk.corpus import stopwords

from nltk.tokenize import TweetTokenizer


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()


# In[26]:


def clean_tweets(txt):
    txt = re.sub(r"\$\w*", " ", txt)
    txt = re.sub(r"^RT\s\w*", " ", txt)
    txt = re.sub(r"#", " ", txt)
    txt = re.sub(r",", " ", txt)
    txt = re.sub(r"(http[s]?:\\/\\/(www\\.)?|ftp:\\/\\/(www\\.)?|www\\.){1}([0-9A-Za-z-\\.@:%_\+~#=]+)+((\\.[a-zA-Z]{2,3})+)(/(.)*)?(\\?(.)*)?", " ", txt)
    txt = re.sub("[0-9]+", " ", txt)
    tokenizer = TweetTokenizer(preserve_case= False, strip_handles = True, reduce_len = True)
    tweet_tokens = tokenizer.tokenize(txt)
    
    tweets_clean = []
    for word in tweet_tokens:
        if  (word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    return tweets_clean
    
dframe["tweets_clean"] = dframe['remove_http'].apply(lambda x: clean_tweets(x))


# In[27]:


dframe


# In[28]:


def remove_punt(tweets):
    tweets = " ".join([char for char in tweets if char not in string.punctuation])
    return tweets
dframe["CLEAN_tweets"] = dframe['tweets_clean'].apply(lambda x: remove_punt(x))


# In[29]:


dframe


# In[30]:


# pip install xlwt


# In[31]:


dframe.sort_values("CLEAN_tweets", inplace=True)
dframe.drop_duplicates(subset= "CLEAN_tweets", keep = "first", inplace = True)
dframe.to_excel("clean_tweets_1.xls", encoding = "utf-8", index =False)


# In[32]:


dframe1 = pd.DataFrame(dframe[["CLEAN_tweets","likes"]])


# In[38]:


dframe1.duplicated().sum()


# In[33]:


dframe1.to_csv('E:\\UNT DATA SCIENCE\INFO 5810\Project\\clean_tweets111.csv')
print('saved to file')


# In[40]:


dframe1


# In[39]:


dframe1.shape


# In[43]:


pip install natsort


# In[48]:


dt = dframe1.groupby("likes").size()
dt.head()


# In[47]:


import seaborn as sns
sns.lineplot(err_style="bars", data =dt)


# In[ ]:




