import nltk, re #operator, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
# from os import listdir
# from os.path import isfile, isdir, join
import numpy
import sys, csv, codecs# getopt, codecs, tim/e, os, csv
from collections import Counter


chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']

def stem(word):
   regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
   stem, suffix = re.findall(regexp, word)[0]
   return stem

def read_bagofwords_dat(myfile):
  bagofwords = numpy.genfromtxt('myfile.csv',delimiter=',')
  return bagofwords

def find_bigrams(input_list):
    return [a + b for (a,b) in zip(input_list, input_list[1:])]

def tokenize_corpus(train=True):
    porter = nltk.PorterStemmer() # also lancaster stemmer
    wnl = nltk.WordNetLemmatizer()
    stopWords = stopwords.words("english")
    docs = []
    classes = []
    positive_words = Counter()
    negative_words = Counter()

    for line in sys.stdin:
        theclass = line.rsplit()[-1]
        raw = line.decode('latin1')
        raw = ' '.join(raw.rsplit()[1:-1])
        # remove noisy characters; tokenize
        raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
        tokens = word_tokenize(raw)
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if w not in stopWords]
        tokens = [wnl.lemmatize(t) for t in tokens]
        tokens = [porter.stem(t) for t in tokens]  
        tokens = Counter(tokens + find_bigrams(tokens))
        docs.append(tokens)
        if int(theclass) == 1: positive_words += tokens 
        else: negative_words += tokens 
        classes.append(theclass)
    return docs, classes, positive_words, negative_words

def find_wordcounts(docs, vocab):
   bagofwords = numpy.zeros(shape=(len(docs),len(vocab)), dtype=numpy.uint8)
   vocabIndex={}
   for i in range(len(vocab)):
      vocabIndex[vocab[i]]=i

   for i in range(len(docs)):
       doc = docs[i]

       for t in doc:
          index_t=vocabIndex.get(t)
          if index_t>=0:
             bagofwords[i,index_t]=bagofwords[i,index_t]+1

   # print "Finished find_wordcounts for:", len(docs), "docs"
   return(bagofwords)

docs, classes, positive_words, negative_words = tokenize_corpus()

vocabfile = open('vocab.txt', 'r')
vocab = [line.rstrip('\n') for line in vocabfile]
vocabfile.close()

outfile= open('data/test_classes.txt', 'w')
outfile.write("\n".join(classes))
outfile.close()

with open("data/bow_test.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(find_wordcounts(docs, vocab))
outfile = open("test_classes.txt", 'w')