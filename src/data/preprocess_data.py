import re
from nltk.stem.porter import *
from nltk.stem.snowball import *

# stemmer = SnowballStemmer("french")
# stemmer = SnowballStemmer("english")
stemmer = PorterStemmer()
stemmer2 = SnowballStemmer("english")
def preprocess(text_string):
	space_pattern = '\s+'
	giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
		'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
	mention_regex = '@[\w\-]+'
	retweet_regex = '^[! ]*RT'
	ds_regex = '√©'
	parsed_text = re.sub(space_pattern, ' ', text_string)
	parsed_text = re.sub(giant_url_regex, '', parsed_text)
	parsed_text = re.sub(mention_regex, '', parsed_text)
	parsed_text = re.sub(retweet_regex, '', parsed_text)
	parsed_text = re.sub(ds_regex, '', parsed_text)
	stemmed_words = [stemmer.stem(word) for word in parsed_text.split()]
	parsed_text = ' '.join(stemmed_words)
	return parsed_text

# def preprocess2(text_string):
# 	space_pattern = '\s+'
# 	giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
# 		'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
# 	mention_regex = '@[\w\-]+'
# 	retweet_regex = '^[! ]*RT'
# 	ds_regex = '√©'
# 	parsed_text = re.sub(space_pattern, ' ', text_string)
# 	parsed_text = re.sub(giant_url_regex, '', parsed_text)
# 	parsed_text = re.sub(mention_regex, '', parsed_text)
# 	parsed_text = re.sub(retweet_regex, '', parsed_text)
# 	parsed_text = re.sub(ds_regex, '', parsed_text)
# 	stemmed_words = [stemmer.stem(word) for word in parsed_text.split()]
# 	parsed_text = ' '.join(stemmed_words)
# 	return parsed_text

# EXAMPLE TWEET
print (preprocess('@user dekoi tu parle jte dis ta ps honte de dire half black lieu de half pakpak attard√©'))