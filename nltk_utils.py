import nltk
import numpy as np
#had to switch the interpreter to the 'pytorch' named environment
#download package from nltk
#nltk.download('punkt')

#import a stemmer:
from nltk.stem.porter import PorterStemmer #there's other stemmers available
stemmer = PorterStemmer()

#creating a method
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]  #tokenized sentence with words
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]  #collected words to check if in sentence
    bow   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0] #place a 'true' or 1 where there are words
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence] #apply stemming to our words 'w' in tokenized sentences
    bag = np.zeros(len(all_words), dtype = np.float32)
    for idx, w, in enumerate(all_words): #loop over all words
        if w in tokenized_sentence:
            bag[idx] = 1.0 #giving a 1 to words in tokenized_sentence

    return bag

#testing out bag of words
# sentence = ["hello", "how", "are", "you"]  
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"] 
# bow   = bag_of_words(sentence, words) #sentences first, words second
# print(bow)
# sentence = ["Tell", "me", "a", "joke"]  
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool", "joke", "funny"] 
# bow   = bag_of_words(sentence, words) #sentences first, words second
# print(bow)

#testing out tokenization
# a = "How long does shipping take?"
# print(a)
# a = tokenize(a)
# print(a)

#testing out the stemming:
# words = ["Organize", "organizes", "organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)

