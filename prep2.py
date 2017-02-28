import nltk, re #operator, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
# from os import listdir
# from os.path import isfile, isdir, join
import numpy
import sys, csv# getopt, codecs, tim/e, os, csv
from collections import Counter

# for l in sys.stdin:
#     exclaim = l.count('!')
#     l = re.split('[\s\.]', l[:-2].lstrip().rstrip())
#     l = [x for x in l[1:] if x]
#     meanwordsize = sum(len(w) for w in l)/len(l)
#     count = len(l)
#     uppers = sum([word.isupper() and len(word) > 2 for word in l])
#     # numbers = sum([word.isnumeric() for word in l])
#     for i in [uppers] :#, exclaim]:
#         print i,
#     print

chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']

def stem(word):
   regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
   stem, suffix = re.findall(regexp, word)[0]
   return stem

def read_bagofwords_dat(myfile):
  bagofwords = numpy.genfromtxt('myfile.csv',delimiter=',')
  return bagofwords

def getPD(positive_words, negative_words, n=5):
    total = positive_words + negative_words
    positive_words.subtract(negative_words)
    best = {k: float(positive_words[k])/total[k] for k in total.viewkeys() & positive_words if int(total[k]) > n}
    pd = sorted(best.items(), key=lambda x: -abs(x[1]))
    return pd

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
        tokens = Counter(tokens)
        # if train == True:
        #     for t in tokens: 
        #         try: words[t] = words[t]+1
        #         except:
        #             words[t] = 1
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

   print "Finished find_wordcounts for:", len(docs), "docs"
   return(bagofwords)

docs, classes, positive_words, negative_words = tokenize_corpus()
pd = getPD(positive_words, negative_words)
print(pd)
vocab = [x[0] for x in pd if abs(x[1]) > .35]
print(vocab)
print(len(pd), len(vocab))
with open("data/BOW.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(find_wordcounts(docs, vocab))
outfile = open("bow_classes.txt", 'w')
outfile.write("\n".join(classes))
outfile.close()
