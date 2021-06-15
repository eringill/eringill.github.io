---
layout: page
title: Natural Language Processing of OK Cupid Profile Data
subtitle: 
---
Data Source
-----------
The data for this project is downloadable from the Codecademy website as a [large .csv file](https://www.codecademy.com/paths/data-science/tracks/dscp-machine-learning-portfolio-project/modules/dscp-group-project-okcupid-date-a-scientist/informationals/dscp-group-project-okcupid-date-a-scientist). The file contains anonymized data in the form of text from over 59 thousand user profiles. The data include age, sex, sexual orientation, location and several essays.

Motivation
----------
I want to explore this data to see whether it is possible to divide users into clusters whose profiles are similar. I'm not so interested in sex, age, sexual orientation, location or any other type of data that would be somewhat easy to cluster. Instead, I want to focus on the words that users employ to describe themselves and their would-be partners. Each user has the opportunity to write 10 essays in their profile, and this is the data that I explore to ask the following:

- How many clusters do the profiles fall into? Are there several possible 'optimal' cluster numbers?
- Which words are most represented in each of these clusters?
- Are these words shared between clusters or are they unique?
- What do the words say about the individuals writing the profiles?

Methods
-------
- The text from all essays written from each individual was concatenated then de-noised and normalized. The TF-IDF algorithm employed to get a measure of word importance in each profile, then the results were clustered via the K-means algorithm. The elbow method suggests optimal cluster numbers of 12 or 16 (12 was chosen for simplicity)
- After fitting the data into 12 clusters, the 10 most common words from each cluster were found and a venn diagram was created to determine overlap between clusters
- The distribution of the 10 most common words from all clusters was then plotted as a heatmap to determine which words were present in most clusters and which words were more rare

Results Summary
---------------
- The words "love", "good" and "really" are among the top 10 in all 12 clusters
- Some clusters have substantial overlap and share up to 7 of their top 10 words, while others contain up to three unique words
- For example:
> - Clusters 10 and 11 share the words "time", "make", "love", "good", "music", "like" and "thing"
> - Only cluster 5 contains the words "travel", "try" and "new"
> - Only cluster 6 contains the words "eye", "want" and "ask"
- Only cluster 3 contains a word that can be unequivocally associated with gender ("guy")
- Some clusters seem to give more clues about the passtimes of members than others, as with cluster 5 one could imagine that these individuals enjoy travel and new experiences, while the words "play", "sport", "movie" and "work" come up in cluster 9. 

Next Steps
----------
- It would be intriguing to see whether genders or ages are overrepresented in any of these clusters. One would expect that these variables are somewhat associated with word choice, but it is not clear how much from this analysis.

Analysis
--------
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
#from collections import Counter
import re
from part_of_speech import get_part_of_speech
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans 
import numpy as np
from supervenn import supervenn


```

This dataset contains text from the profiles of over 59,000 OK Cupid users. I want to explore this data to see whether it is possible to divide the users up into clusters whose profiles are similar. I'm not so interested in sex, age, sexual orientation, location or any other type of data that would be somewhat easy to cluster. Instead, I want to focus on the **words** that users employ to describe themselves and their would-be partners. Each user has the opportunity to write 10 essays in their profile, and this is the data that I'm going to explore to ask the following:
- How many clusters do the profiles fall into? Are there several possible 'optimal' cluster numbers?
- Which words are most represented in each of these clusters?
- Are these words shared between clusters or are they unique?
- What do the words say about the individuals writing the profiles?  

First I'll read the file containing the data, clean and normalize the text.


```python
#read csv file containing profile data as DataFrame
df = pd.read_csv('../okcupiddata/profiles.csv')
#make dataframe just containing essays
df_essays = df[['essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8', 'essay9']]
#join all of the columns together into a new column and convert to list
df_essays['corpus'] = df_essays[df_essays.columns[0:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1)
corpus = df_essays['corpus'].tolist()

#function that removes noise from text
def remove_noise(text):
    '''
    input:
    text- string of text
    output: 
    string of text with html tags, punctuation, newline and tab chars removed
    '''
    #remove html tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = re.sub('href', '', text)
    text = re.sub('classilink', '', text)
    text = re.sub('ilink', '', text)
    text = re.sub('&amp', '', text)
    #remove punctuation
    text = re.sub(r"[^\w\d'\s]+",'',text)
    #remove newline, tab
    text = re.sub("(\\d|\\W)+"," ",text)
    return text

#function that lemmatizes text, removes stop words and words <3 characters
def lemmatize(text):
    '''
    input:
    text- string of text
    output:
    lemmatized string of text with stop words and words <3 chars removed
    '''
    #instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    #tokenize the string for lemmatization and stop word removal
    tokenized = word_tokenize(str(text))
    lemmatized = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized if not token in stop_words and len(token) >2]
    #re-join the tokenized text so that it can be used for Tfidf
    cleaned = " ".join(lemmatized)
    return cleaned

#remove noise from text and lemmatize
denoised = [remove_noise(str(i)) for i in corpus]
normalized = [lemmatize(i) for i in denoised]

```


Next I'll employ the TF-IDF algorithm to get a measure of word importance in each profile. Then I'll cluster the TF-IDF results using the K-means algorithm so that profiles with similar word importance metrics are grouped together. 


```python
#instatiate Tfidf vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
#fit and transform profile data with vectorizer
tfidf = vectorizer.fit_transform(normalized)

#set up elbow method to determine optimal number of clusters for K-means algorithm
Sum_of_squared_distances = []
#test cluster numbers ranging from 2 to 19
K = range(2,20)

for k in K:
   km = KMeans(n_clusters=k, max_iter=200, n_init=10)
   km = km.fit(tfidf)
   Sum_of_squared_distances.append(km.inertia_)

#plot out the sum of squared distances for each K   
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
#12 could work or 16
```


    
![svg](/assets/img/date-a-scientist_files/date-a-scientist_5_0.svg)
    


It looks like I could select 12 or 16 as optimal numbers of clusters. For the sake of simplicity, I'm going to select 12.


```python
true_k = 12
#set up clustering
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=true_k)
#fit the data
model.fit(tfidf)

#this loop transforms the numbers back into words
common_words = model.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words):
    #print out the top 10 words in each cluster
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
```

    0 : class, like, love, thing, people, good, time, make, music, think
    1 : like, people, thing, good, think, really, love, time, know, make
    2 : lol, love, like, friend, know, good, work, time, life, movie
    3 : friend, work, fun, like, family, good, look, love, life, guy
    4 : love, friend, good, like, music, life, movie, people, family, food
    5 : new, friend, love, enjoy, travel, like, good, work, try, time
    6 : ask, work, like, want, know, look, eye, love, friend, good
    7 : life, love, enjoy, like, good, friend, people, time, make, live
    8 : sport, friend, like, love, play, work, family, good, movie, enjoy
    9 : music, movie, love, food, book, like, friend, good, work, make
    10 : like, love, make, thing, work, music, good, people, time, friend
    11 : like, thing, love, good, really, music, make, time, movie, think


It looks like there is some overlap between the top most important words in each cluster. Now I'll plot a modified venn diagram to visualize this overlap. For orientation: 
- The number of clusters that each group of words is present in is on the top of the diagram. 
- The number of words that I've used to make the diagram is on the right of the diagram. 
- The number of words in the set of words that is between the vertical lines is on the bottom of the diagram.


```python
#make a large set of the top ten words from each cluster 
word_list = []
for centroid in common_words:
    word_list.append(set(words[word] for word in centroid))

#plot a modified venn diagram so we can see how the words are shared between clusters
plt.figure(figsize=(16, 8))
supervenn(word_list)
```




    <supervenn._plots.SupervennPlot at 0x7fcf868e71c0>




    
![svg](/assets/img/date-a-scientist_files/date-a-scientist_9_1.svg)
    



```python
#make a set of all of the words that are in the top 10 lists for each cluster
word_set = []
for cluster in word_list:
    word_set = word_set + list(cluster)

word_set = list(set(word_set))

```

    33



```python
#make a dataframe that indicates whether each of the words in the set are in each cluster
new_list = []
for cluster in word_list:
    lst = []
    for word in word_set:
        if word in cluster:
            lst.append(True)
        else:
            lst.append(False)
    new_list.append(lst)

word_df = pd.DataFrame(new_list, columns = word_set)
word_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>work</th>
      <th>time</th>
      <th>lol</th>
      <th>make</th>
      <th>food</th>
      <th>enjoy</th>
      <th>class</th>
      <th>look</th>
      <th>love</th>
      <th>really</th>
      <th>...</th>
      <th>sport</th>
      <th>live</th>
      <th>guy</th>
      <th>people</th>
      <th>play</th>
      <th>like</th>
      <th>thing</th>
      <th>ask</th>
      <th>life</th>
      <th>fun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>12 rows × 33 columns</p>
</div>



Now I'm going to visualize each word individually in a heat map. Blue squares indicate that a word is present in a cluster.


```python
#make a heatmap out of the dataframe so we can visualize which words are common and which words 
#are rare among the clusters
word_df_bin = word_df * 1
sns.set_theme()
dims = (15, 8)
fig, ax = plt.subplots(figsize= dims)
ax = sns.heatmap(word_df_bin, cbar = False, cmap= "Blues")
plt.ylabel("cluster")
plt.xlabel("word")
```




    Text(0.5, 48.453125, 'word')




    
![svg](/assets/img/date-a-scientist_files/date-a-scientist_13_1.svg)
    



```python

```
