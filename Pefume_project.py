#!/usr/bin/env python
# coding: utf-8

# # Automatic perfume recommendation program
# 
# 
# #### Ariel Ye eun Kwon - https://www.linkedin.com/in/arielkwon/
# 
# Hi everyone, I'm Ariel, a great perfume enthusiast and senior student at Sungshinn University in Seoul, South Korea, where I'm studying business and data-informatics.
# 
# Not too long ago, my interest with perfumery began with my first scent, Flower by Kenzo, a freebie my mom received during her shopping. Starting small my interest grew, leading to the creation of my perfume collection. With a keen sense of my preferred fragrances and favorite perfume house, I've curated a selection of top picks that often feature shared notes like sandalwood or woody essences.
# 
# This sparked an idea - what if I could use Python to compute similarity scores and develop an automated recommendation system for fellow enthusiasts like myself? The potential highlight? Discovering 'new' and 'unknown' perfumes beyond one's usual choices, yet still resonating with the sought-after scents.
# 
# The world of perfumery is vast and intriguing! In this notebook, I'll delve into the intricate details of how I crafted this program. Let's go!
# 

# In[ ]:





# The dataset is enlisted with over 30,000 products.
# 
# The dataset is consisted with perfume brands, perfume names, and their notes (20 notes maximum), and gender column if that perfume is made for women or men, or unisex. 

# In[63]:


#before we start

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.display import Image


# In[64]:


df=pd.read_csv("C:/Users/ariel/OneDrive/바탕 화면/Python/perfume data.csv", encoding='cp949')


# In[65]:


df.info()


# In[66]:


df.head()


# This dataset is a set of perfumes with their notes / brand / gender

# # 1. Data cleaning

#         
# 1. Missing Rows: According to the output of df.info(), there are no missing rows in the dataset.
# 
# 
# 
# 2. Duplicated Rows: Two minor issues were identified regarding duplicated rows
# 
#         -Perfumes (rows) that are idential duplicates.
#         -Variants of the same perfume that differ in their notes, rendering them unique even though they may initially seems like the same perfume.
# 
# 
# 3. Mislabeled Rows : Some perfumes labeled as for women are actually meant for men, and vice versa, within the dataset.

# ### 1-1. Missing rows

# In[67]:


missing_values = df.isnull().sum()
if missing_values.any():
    print("Columns with missing values:")
    print(missing_values)


# The key columns to focus on are 'brand' and 'title'. It's fine if the 'note' and 'gender' columns have missing values. I am going to handle this problem later on.

# ### 1-2. Duplicated rows

# In[68]:


duplicate_title_rows = df[df.duplicated(subset=['title'])]
if duplicate_title_rows.shape[0] > 0:
    print("Duplicate rows based on 'title' column found:")
    print(duplicate_title_rows)


# solutions : I am going delete the duplicated rows IF the note1 and note 2 columns are the same
# 
# we have 198 rows to tackle on.

# In[69]:


df.drop_duplicates(subset=["title","note1","note2"], keep='first', inplace=True)


# I  have found that the new version of perfume has different notes. Based on that information, I am going to drop the truely duplicated perfumes.
# 

# In[70]:


duplicate_new = df[df.duplicated(subset=['title','note1','note2'])]
if duplicate_new.shape[0] > 0:
    print("Duplicate rows based on 'title' column found:")
    print(duplicate_new)


# In[71]:


duplicate_new.count()


# No duplicated rows. But we would have to rename the updated of each perfumes not to get confused. I found that there are sill 60 perfumes with an update.

# In[72]:


duplicate_title_rows2 = df[df.duplicated(subset=['title'])]
if duplicate_title_rows2.shape[0] > 0:
    print("Duplicate rows based on 'title' column found:")
    print(duplicate_title_rows2)


# In[73]:


second_occurrences = df.duplicated(subset="title", keep='last')


df.loc[second_occurrences, "title"] += " (2nd version)"


# In[74]:


third_occurrences = df.duplicated(subset="title", keep='last')


df.loc[third_occurrences, "title"] += " 3rd version)"


# In[75]:


duplicate_title_rows3 = df[df.duplicated(subset=['title'])]
if duplicate_title_rows3.shape[0] > 0:
    print("Duplicate rows based on 'title' column found:")
    print(duplicate_title_rows3)


# In[76]:


duplicate_title_rows3


# There were some perfumes with 3rd updates so I had to double rename them. Now the duplication process is finally done!

# ### 1-3. Mislabeled rows

# In[77]:


df[df['title'] == 'Santal 33 Le Labo for women and men']  

#EXAMPLE - it is one of my favorite  perfumes!


# Upon examining the titles more closely, it becomes apparent that each perfume is labeled with its name, followed by an indication of whether it is a unisex, women's, or men's perfume. Therefore, I will create a function to extract this gender information and place it under the gender column.

# In[78]:


def extract_gender(info):
    if 'for women and men' in info:
        return 'unisex'
    elif 'for women' in info:
        return 'women'
    elif 'for men' in info:
        return 'men'
    else:
        return 'unknown'  # Default case if none matches


# In[79]:


df['gender'] = df['title'].apply(extract_gender)


# In[80]:


df[df['title'] == 'Santal 33 Le Labo for women and men']


# Now the gender column perfectly fits the title.

# ## 2. TF-IDF Vectorization for Perfume Notes

# In this stage, I will be identifying the similarities among the notes.

# 1. TF-IDF
#     ->cosine similarity / to measure how similar texts or documents are to each other.
# 
# 
# 2. Term frequency TF : number of times of appeared / number of terms in that document
# 
# 
# 3. Inverse document frequency IDF : importance of the term
# 					total number of documents / number of docs 'containing' the term ->logarithm
# 			-> terms that appear in small number of docs are more important

# In[81]:


note_columns = [f'note{i}' for i in range(1, 21)]


# In[82]:


df[note_columns].fillna('', inplace=True)
df[note_columns] = df[note_columns].astype(str)


# In[83]:


df['combined_notes'] = df[note_columns].apply(lambda x: ' '.join(x), axis=1)

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_notes'])


# In[84]:


df.head(2)


# The new column created next to the 'note20' column shows a consolidation of all the notes present in each perfume.

# ## 3. Cosine similarity - within the same gender group

# In[96]:


def recommend_similar_perfumes(input_perfume_name, df, tfidf_matrix, top_n=5):
    if input_perfume_name not in df['title'].values:
        print("Perfume not found...")  # just in case
        return
    
    original_idx = df.index[df['title'] == input_perfume_name].tolist()[0]
    input_gender = df.at[original_idx, 'gender']

    # Gender filter
    gender_filtered_df = df[df['gender'] == input_gender]

    if input_perfume_name not in gender_filtered_df['title'].values:
        print("Perfume not found")    
        return

    filtered_idx = gender_filtered_df.index[gender_filtered_df['title'] == input_perfume_name].tolist()[0]
    tfidf = TfidfVectorizer(stop_words='english')
    gender_filtered_tfidf_matrix = tfidf.fit_transform(gender_filtered_df['combined_notes'])
    
    # Recalculate the correct index in the filtered TF-IDF matrix
    correct_idx = list(gender_filtered_df.index).index(filtered_idx)
    cosine_sim = cosine_similarity(gender_filtered_tfidf_matrix[correct_idx], gender_filtered_tfidf_matrix).flatten()
    
    cosine_sim[correct_idx] = 0   # Exclude the input perfume
    
    # Sort the perfumes based on their cosine similarity scores and get top recommendations
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    recommended_perfumes = gender_filtered_df.iloc[top_indices]['title']
    
    return recommended_perfumes


# Name: recommend_similar_perfumes
# 
# Parameters:
# 
# #### input_perfume_name: The name of the input perfume for similar recommendations.
# 
# #### df: The dataframe containing information about perfumes, including columns such as 'title', 'gender', and 'combined_notes'.
# 
# #### tfidf_matrix: The TF-IDF matrix used for calculating the cosine similarity between perfumes based on their combined notes.
# 
# #### top_n: The number of top similar perfumes to recommend. By default, set to 5.

# # 4. Result

# In[97]:


input_perfume_name = 'Santal 33 Le Labo for women and men'  #one of my favorite perfumes 

recommendations = recommend_similar_perfumes(input_perfume_name, df, tfidf_matrix, top_n=5)
print("Top recommendations similar to", input_perfume_name, "\n", recommendations)


# In[42]:


pd.DataFrame(recommendations)


# The output seems to be similar based on the perfume notes.

# In[43]:


df[df['title']=='Santal 33 Le Labo for women and men']


# In[45]:


df[df['title']=="Santal's Kiss Alexandria Fragrances for women and men"]


# ### I've noticed some shared notes between the two perfumes; I'm going to assess these results further on

# In[88]:


input_perfume_name2 = 'Baccarat Rouge 540 Maison Francis Kurkdjian for women and men' #masterpiece !

recommendations = recommend_similar_perfumes(input_perfume_name2, df, tfidf_matrix, top_n=5)
print("Top recommendations similar to", input_perfume_name, "\n", recommendations)


# # 5. Evaluating the result

# 1. Python : The diversity code involves systematically assessing the similarities among the recommended perfumes. A low score indicates a greater likeness is desirable, with the target being less than 0.3.
# 
# 
# 2. Maunally : Engaging in manual review within the esteemed perfume community Fragrantica, I will personally evaluate the resemblance of the recommended perfumes to the input.

# ## 5-1) Using the diversity score

# To assess the accuracy of the recommendation system, we can compare the similarities between the recommended perfumes. The "diversity score" is a metric that quantifies how similar the set of recommended perfumes is to each other.
# 
# A lower diversity score indicates that the recommended perfumes are more alike to each other, while a higher score suggests greater diversity.
# 
# By using this technique, we can evaluate the fairness of the recommendation results. It allows us to scrutinize the extent of similarity within the recommended perfume set.

# In[98]:


def calculate_diversity(input_perfume_name, df, tfidf_matrix):
    idx = df.index[df['title'] == input_perfume_name].tolist()[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    top_indices = cosine_sim.argsort()[-6:-1]
    top_similarities = cosine_sim[top_indices]  # The similarity between the recommendations
    
    diversity = np.std(top_similarities)  # Standard deviation for precise result
    return diversity

diversity_score = calculate_diversity("Santal 33 Le Labo for women and men", df, tfidf_matrix)
print("Diversity Score: ", diversity_score)


# In[99]:


def calculate_diversity(input_perfume_name, df, tfidf_matrix):
    idx = df.index[df['title'] == input_perfume_name].tolist()[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    top_indices = cosine_sim.argsort()[-6:-1]  
    top_similarities = cosine_sim[top_indices]  #the similarity between the recommendations
    
    diversity = np.std(top_similarities)   #standard deviation for precise result
    return diversity

diversity_score = calculate_diversity('Baccarat Rouge 540 Maison Francis Kurkdjian for women and men', df, tfidf_matrix)
print("Diversity Score: ", diversity_score)


# Seems they are pretty close to zero 

# ## 5-2) References in Fragrantica

# In addition to reviewing the results in Python code, I will cross-reference the outcomes manually on Fragrantica to ascertain whether users have aligned their votes similarly to the results derived from the Python code.

# In[103]:


Image("C:/Users/ariel/OneDrive/바탕 화면/Python/dupe.png")

# the first recommendation (with high similarity score with Santal 33)


# https://www.fragrantica.com/perfume/Alexandria-Fragrances/Santal-s-Kiss-47944.html
# 
# Resourse : Fragrantica -> It's evident that individuals have voted for Santal's Kiss depicting a similar fragrance profile to the input provided. (Santal 33)

# In[102]:


Image("C:/Users/ariel/OneDrive/바탕 화면/Python/dupe2.png")

# the first recommendation (with high similarity score with Baccarat Rouge 540)


# https://www.fragrantica.com/perfume/Shakespeare-Perfumes/Julius-Caesar-54082.html
# 
# A renowned perfume, Baccarat Rouge 540 by MFK. Users on Fragrantica have also noted similarities between this fragrance and Julius Caesar by Shakespeare Perfumes.

# # FINISH

# limitations
# 
# -It is unknown if some perfumes are discontinued to be made. And brand new perfumes might not be enlisted.
# 
# 
# -Some niche perfums are only sold at specific countries and it might be hard to get one 
# -> in this case you can amplify the recommendation number by 10 or 20 for accessibility.

# ### THANK YOU

# In[ ]:




