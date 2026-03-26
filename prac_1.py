import nltk
from nltk.tokenize import word_tokenize, WhitespaceTokenizer, TreebankWordTokenizer, TweetTokenizer, MWETokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer


# User input
text = input("Enter a sentence: ")

print("\nOriginal Text:", text)

# ---------------- TOKENIZATION ----------------

# 1. Whitespace Tokenization
ws = WhitespaceTokenizer()
print("\nWhitespace Tokenization:", ws.tokenize(text))

# 2. Punctuation-based Tokenization
tokens = word_tokenize(text)
print("\nPunctuation Tokenization:", tokens)

# 3. Treebank Tokenization
tb = TreebankWordTokenizer()
print("\nTreebank Tokenization:", tb.tokenize(text))

# 4. Tweet Tokenization
tt = TweetTokenizer()
print("\nTweet Tokenization:", tt.tokenize(text))

# 5. MWE Tokenization
mwe = MWETokenizer([('machine', 'learning'), ('New', 'York')])
print("\nMWE Tokenization:", mwe.tokenize(word_tokenize(text)))

# ---------------- STEMMING ----------------

ps = PorterStemmer()
ss = SnowballStemmer("english")

print("\nPorter Stemming:")
for word in tokens:
    print(word, "->", ps.stem(word))

print("\nSnowball Stemming:")
for word in tokens:
    print(word, "->", ss.stem(word))

# ---------------- LEMMATIZATION ----------------

lemmatizer = WordNetLemmatizer()

print("\nLemmatization:")
for word in tokens:
    print(word, "->", lemmatizer.lemmatize(word))