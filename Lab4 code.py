#Maaz Muhammad Khan
#2788572

import os.path
import scipy.spatial.distance
from scipy import spatial
import nltk, string, numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from urllib.request import urlopen
import re
from nltk.stem import PorterStemmer
from nltk import FreqDist
from nltk import bigrams
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

def tasks_execution():
    
    phrase_list = ["data mining",        "machine learning","deep learning"]   
    keyword_list = ["research"," data", "mining", "analytics",]

    dl=["https://en.wikipedia.org/wiki/Engineering",
"http://my.clevelandclinic.org/research",
"https://en.wikipedia.org/wiki/Data_mining",
"https://en.wikipedia.org/wiki/Data_mining#Data_mining",
"http://cis.csuohio.edu/~sschung/"]
    count_matrix = []
    md=[]
#document vector construction
    for document in dl:
        file_name = "./" + document.replace("/", "_").replace(":","_").replace("?","_") + ".txt"
        print ("Document Link: ",document)
        if not os.path.exists(file_name):
            f = open(file_name, 'wb')

            f.write(text_extraction(document).lower())
            f.close()

#calculation of keyword frequency
        f = open(file_name, encoding="utf8")
        row = f.read()
        #token = row.split()
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        token = tokenizer.tokenize(row)
          
        stop_words = set(stopwords.words('english'))           
        filtered_sentence = [w for w in token if not w in stop_words] 
          
        filtered_sentence = [] 
          
        for w in token: 
            if w not in stop_words: 
                filtered_sentence.append(w) 

        phrase = FreqDist(bigrams(filtered_sentence))
        record = []
        d1=[]
        x1=[]
        for val in phrase_list:
            k, v = val.split()
            count = phrase.get((k, v))
            if count is None:
                count = 0
            record.append({val.replace(" ", "_"): count})
            d1.append(count)
        # frequency of words matrix
        token_freq = FreqDist(token)
        for word in keyword_list:
            word_count = token_freq[word]
            # remove duplicate count for word in phrase
            for val in phrase_list:
                if word in val.split():
                    phrase_key = val.replace(" ", "_")
                    word_count = word_count - record[0][phrase_key]
            record.append({word: word_count})
            d1.append(word_count)
        print("x",x1)  
        print ("Record :", record)
        count_matrix.append(record)
        md.append(d1)
    print ("                  Document Vectors")
    print("------------------------------------------------------ ")
    print ("Count Matrix :", count_matrix)
    print(md)
    from sklearn.metrics.pairwise import cosine_similarity
    #x=cosine_similarity(l, l)
    print ("\n*************Document  Vectors  For Keywords and Phrase List***********")
    print("------------------------------------------------------ ")
    print("data mining|machine learning|deep learning|research|data|mining|analytics")
    print("------------------------------------------------------ ")
    for i in range(len(md)):
       print("Document",i+1,md[i])
    print ("\n***************************Cosine Similarity*******************")
    csm=np.zeros((len(md),len(md)))
    cosine_similiarity1=0
    for i in range(len(md)):
        for j in range(len(md)):
            cosine_similiarity1 =1-spatial.distance.cosine(md[i], md[j])
            csm[j][i]=cosine_similiarity1 
    print("          Document 1 Document 2 Document 3 Document 4 Document 5")
    print("          ------------------------------------------------------ ")
    for i in range(len(csm)):
       print("Document",i+1,csm[i])
def text_extraction(document):
   
    website = urlopen(document)
    html = website.read()
    soup = BeautifulSoup(html, "html.parser")
    html_tag = soup.html

    for link in soup("link"):
        link.decompose()

  
    for script in soup("script"):
        script.decompose()

    ps = PorterStemmer()
    text = ""
    for w in re.sub(' +',' ',html_tag.get_text()).split(" "):
        text += ps.stem(w) + " "
    return text.encode("utf-8")



























tasks_execution()