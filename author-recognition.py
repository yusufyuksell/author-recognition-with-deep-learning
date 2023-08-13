import numpy as np
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from snowballstemmer import TurkishStemmer
from gensim.models import KeyedVectors
from sklearn import preprocessing
import glob
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

    
word_vectors = KeyedVectors.load_word2vec_format('trmodel', binary=True)
model = Word2Vec(sentences=common_texts, vector_size=100,
                 window=5, min_count=1, workers=4)




path = 'Yazarlar'
authors = glob.glob(path + "/*")

all_files = glob.glob(path + "/*.txt")
count = 0
encodedtext = np.zeros((2000,400))

label = ()
for i in range(1, 21):
    label += (i,) * 100

    
for h in range(20):
    for text in glob.glob(authors[h], recursive=True):
            
        
       all_files = glob.glob(text + "/*.txt")
    for i in all_files:
        punctuiations = "!#$%&'()*+,-./:’;<=>?@[\]^_`{|}~'"
        total = np.zeros((1,400))
        textfile = open(i,'r',encoding="utf-8")
        str = textfile.read()
        no_punct = " "
        
        for char in str:
            if(char not in punctuiations):
                no_punct = no_punct + char
        stop_words = stopwords.words('turkish')
        tokenize_words = word_tokenize(no_punct)
        tokenize_words_without_stopwords = []
        
        for word in tokenize_words:
            if word not in stop_words:
                tokenize_words_without_stopwords.append(word)
        stammer = TurkishStemmer()
        input_str = tokenize_words_without_stopwords

        
        
        for word in input_str:
            try:
                total += (word_vectors.get_vector(stammer.stemWord(word).lower()))
                
            except:
                continue
        
        
        total = preprocessing.normalize(total)
        total = total.reshape(400)
        encodedtext[count,:] = total
        count += 1    
        
        
    
    
    arr = np.column_stack((encodedtext, label))        
    
    shuffled_list = np.copy(arr)
    np.random.shuffle(shuffled_list)
    
    all_but_last_column = shuffled_list[:, :-1]
    last_columns = shuffled_list[:, 400:]
   
    
# Decision tree classifier
    from sklearn.tree import DecisionTreeClassifier
    X_dec = all_but_last_column
    y_dec = last_columns
    
    
    X_dec_train, X_dec_test, y_dec_train, y_dec_test = train_test_split(X_dec, y_dec, 
                                                       test_size = 0.3, random_state = 0)
    
    
    
    dtree_model = DecisionTreeClassifier(max_depth = 10).fit(X_dec_train, y_dec_train)
    tahmin_dtree = dtree_model.predict(X_dec_test)
    
    accuracy_dec = accuracy_score(y_dec_test,tahmin_dtree )
    mse_dec = mean_squared_error(y_dec_test, tahmin_dtree)
    cm_dec = confusion_matrix(y_dec_test, tahmin_dtree)
    
    
# SVM 
    from sklearn.svm import SVC
    X_svm = all_but_last_column
    y_svm = last_columns
  
    X_svm_train, X_svm_test, y_svm_train, y_svm_test = train_test_split(X_svm, y_svm, 
                                                       test_size = 0.3, random_state = 0)
  
    
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_svm_train, y_svm_train)
    tahmin_svm = svm_model_linear.predict(X_svm_test)
  
    accuracy_svm = svm_model_linear.score(X_svm_test, y_svm_test)
    mse_svm = mean_squared_error(y_svm_test, tahmin_svm)
    cm_svm = confusion_matrix(y_svm_test, tahmin_svm)


# KNN
    from sklearn.neighbors import KNeighborsClassifier
    X_knn = all_but_last_column
    y_knn = last_columns
  
    X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(X_knn, y_knn, 
                                                       test_size = 0.3, random_state = 0)
  
    knn = KNeighborsClassifier(n_neighbors = 10).fit(X_knn_train, y_knn_train)
  
    accuracy_knn = knn.score(X_knn_test, y_knn_test)
    tahmin_knn = knn.predict(X_knn_test) 
    mse_knn = mean_squared_error(y_knn_test, tahmin_knn)
    cm_knn = confusion_matrix(y_knn_test, tahmin_knn)


# GNB
    from sklearn.naive_bayes import GaussianNB
    X_gnb = all_but_last_column
    y_gnb = last_columns
  
    X_gnb_train, X_gnb_test, y_gnb_train, y_gnb_test = train_test_split(X_gnb, y_gnb,
                                                       test_size = 0.3, random_state = 0)
  
    
    gnb = GaussianNB().fit(X_gnb_train, y_gnb_train)
    tahmin_gnb = gnb.predict(X_gnb_test)
  
    accuracy_gnb = gnb.score(X_gnb_test, y_gnb_test)
    mse_gnb = mean_squared_error(y_gnb_test, tahmin_gnb)
    cm_gnb = confusion_matrix(y_gnb_test, tahmin_gnb)    
    
    
    
    
    
np.savetxt('sonuc.npy', arr, delimiter="\t")