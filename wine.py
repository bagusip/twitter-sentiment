import csv
import math
import numpy as np
import re
import nltk
import pandas as pd
import decimal

from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

# clear predict file 
filename = "predict.csv"
f = open(filename,"w+")
f.close()

# inisiasi sastrawi&preprocessing
stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocessing(tweets):
  tweet = []
  tweet = tweets[1:]
  #jadiin tipe data string
  tweet_string = ''.join(map(str, tweets))
  #stopword removal
  stop = stopword.remove(tweet_string)
  #case folding
  lower = stop.lower()
  #filtering
  kalimat = re.sub(r'@[^\s]+','',lower)
  kalimat2 = re.sub("http\S+", "link",kalimat) 
  kalimat3 = re.sub('\d+','',kalimat2)
  
  #stemming
  hasil = stemmer.stem(kalimat3)
  #tokenizing
  tokenize = nltk.tokenize.word_tokenize(hasil)
  return tokenize
#baca dataset train dan text-preprocessing
print('pre processing start')
with open('train.csv', 'r', encoding="utf8") as f:
    reader = csv.reader(f, dialect='excel', delimiter=';')
    next(reader, None)
    tweets = [[preprocessing(row[0]),int(float(row[1]))] for row in reader]

#Shuffles our rows. This allows us to construct (roughly) a training dataset
np.random.shuffle(tweets)

#Creates a training dataset e.g. [ [tweets, manual_sentiment], ... ]
# training = tweets[1:500]
#remove header
training = tweets[1:]

#Defines classes: 0 is negative, 1 is positive
classes = [0, 1]

def train_naive_bayes(training,classes):
    #Initialize D_c[ci]: list semua dokumen untuk kelas i
    #E.g. D_c[1] adalah list dari [tweets, sentiments] utk kelas i

    D_c = [[]] * len(classes)

    #Initialize n_c[ci]: jumlah dokumen untuk kelas i
    n_c = [None] * len(classes)

    #Initialize logprior[ci]: menyimpan prior probability untuk kelas i
    logprior = [None] * len(classes)

    #Initialize loglikelihood: loglikelihood[ci][wi] menyimpan likelihood probability untuk wi(word) given class i
    loglikelihood = [None] * len(classes)

    #Partisi dokumen menjadi 2 kelas. D_c[0]: negative docs, D_c[1]: positive docs
    for obs in tweets: 
        if obs[1] == 1:
            D_c[1] = D_c[1] + [obs]
        elif obs[1] == 0:
            D_c[0] = D_c[0] + [obs]

    #Buat vocabulary list. 
    V = []
    counter = 0
    for obs in tweets:
      counter+=1
      for word in obs[0]:
          counter+=counter
          if word in V:
            continue
          else:
            V.append(word)

    V_size = len(V)

    #n_docs: jumlah dokumen pada training set
    n_docs = len(tweets)

    for ci in range(len(classes)):
        #menyimpan n_c value utk setiap kelas
        n_c[ci] = len(D_c[ci])

        #Compute P(c) -> probabilitas terklasifikasinya suatu dokumen ke dalam suatu kategori
        # menggunakan log untuk menghindari floating-point underflow
        logprior[ci] = np.log((n_c[ci] + 1)/ n_docs)
        print(logprior)
        #Counts total number of words in class c
        #menghitung jumlah kata di class c
        count_w_in_V = 0

        for d in D_c[ci]:
            count_w_in_V = count_w_in_V + len(d[0])
        denom = count_w_in_V + V_size
        dic = {}
        #Hitung P(w|c)
        for wi in V:
            #Count number of times wi appears in D_c[ci]
            #Hitung berapa kali wi muncul pada D_c[ci]
            count_wi_in_D_c = 0
            for d in D_c[ci]:
                for word in d[0]:
                    if word == wi:
                        count_wi_in_D_c = count_wi_in_D_c + 1
            numer = count_wi_in_D_c + 1
            # frekuensi kemunculan suatu kata dalam suatu dokumen dalam suatu kategori
            # menggunakan log untuk menghindari floating-point underflow
            dic[wi] = np.log((numer) / (denom))
        loglikelihood[ci] = dic
    
    
    with open('test.csv', 'r', encoding="utf8") as f:
      reader = csv.reader(f, dialect='excel', delimiter=';')
      next(reader, None)
      for row in reader:
        test_naive_bayes(row[0],row[1],preprocessing(row[0]),logprior,loglikelihood,V) 
    
    

    return V,logprior,loglikelihood

def test_naive_bayes(raw_testdoc,raw_sentiment,testdoc, logprior, loglikelihood, V):
  #Initialize logpost[ci]: stores the posterior probability for class ci
    prediksi = []
    logpost = [None] * len(classes)
    with open('predict.csv', mode='a', encoding="utf8") as f:
      f = csv.DictWriter(f, fieldnames = ["aktual","prediksi"])
    # sumloglikelihoods = 0
      for ci in classes:
          sumloglikelihoods = 0
          for word in testdoc:
              if word in V:
                  #This is sum represents log(P(w|c)) = log(P(w1|c)) + log(P(wn|c))
                  sumloglikelihoods += loglikelihood[ci][word]
      #Computes P(c|d)
          logpost[ci] = logprior[ci] + sumloglikelihoods
        #Return the class that generated max ĉ
      prediksi = logpost.index(max(logpost))
      print(testdoc)
      print(prediksi)
      # print('aktual : ', raw_sentiment,', prediksi : ', prediksi)
      f.writerow({'aktual': raw_sentiment,'prediksi': prediksi})
      # test(prediksi)
     
      return logpost.index(max(logpost))


print('TRAIN START')
train_naive_bayes(training,classes)
df = pd.read_csv('predict.csv',header=None,usecols=[0,1])

df.rename(columns={0: "aktual",1:"prediksi"})
actual = list(df[0])
predicted = list(df[1])
len_train = len(tweets)
len_test = len(actual)
percent_train = decimal.Decimal((len_train/(len_train+ len_test))*100)
percent_test = decimal.Decimal(100-percent_train)
accuracy = decimal.Decimal(accuracy_score(actual, predicted)*100)
print("train set: ", float(str(round(percent_train))),"%")
print(len_train)
print(actual)
print("test set: ", float(str(round(percent_test))),"%")
print(len_test)
print(predicted)
results = confusion_matrix(actual, predicted)
print('Confusion Matrix :')
print(results) 
print('Accuracy Score : ',accuracy,"%")
print('Report : ')
print(classification_report(actual, predicted))
TP = results[1, 1]
TN = results[0, 0]
FP = results[0, 1]
FN = results[1, 0]

precision = TP/ float(TP+FP)
recall = TP/ float(TP+FN)
fmeasure = 2*(recall*precision)/(recall+precision)

print('precision : ',precision)
print('recall : ',recall)
print('fmeasure : ',fmeasure)