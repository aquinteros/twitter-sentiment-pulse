import pandas as pd
import snscrape.modules.twitter as sntwitter
import datetime as dt
from dateutil.relativedelta import relativedelta
import fastparquet
import streamlit as st
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet
import plotly.express as px
from streamlit_extras.buy_me_a_coffee import button as buy_me_a_coffee
from streamlit_extras.mention import mention

st.set_page_config(page_title="Twitter Sentiment Pulse", page_icon=":bird:")

def get_tweets(concepto, start_date, end_date, progressbar, lang = 'es', max_tweets=1_000):
	"""Imports tweets from Twitter API using snscrape module."""

	scrap = sntwitter.TwitterSearchScraper(f'{concepto} since:{start_date} until:{end_date} lang:{lang}').get_items()

	tweets_list = {}

	for i, tweet in enumerate(scrap):
		
		tweets_list[i] = {
			'id': str(tweet.id),
			'datetime': tweet.date,
			'rawContent': tweet.rawContent,
			'replyCount': tweet.replyCount,
			'retweetCount': tweet.retweetCount,
			'likeCount': tweet.likeCount,
			'quoteCount': tweet.quoteCount,
			# 'lang': tweet.lang,
			# 'place': tweet.place,
			'hashtags': tweet.hashtags,
			# 'mentionedUsers': tweet.mentionedUsers,
			'user_id': tweet.user.id,
			'user_name': tweet.user.username,
			# 'user_renderedDescription': tweet.user.renderedDescription,
			# 'user_join_date': tweet.user.created,
			# 'user_followers': tweet.user.followersCount,
			# 'user_location': tweet.user.location,
			'user_verified': tweet.user.verified,
			# 'inReplyToTweetId': str(tweet.inReplyToTweetId)
		}

		dtypes = {
			'id': 'string',
			'datetime': 'datetime64[ns, UTC]',
			'rawContent': 'string',
			'replyCount': 'int64',
			'retweetCount': 'int64',
			'likeCount': 'int64',
			'quoteCount': 'int64',
			# 'lang': 'string',
			# 'place': 'string',
			'hashtags': 'string',
			# 'mentionedUsers': 'string',
			'user_id': 'string',
			'user_name': 'string',
			# 'user_renderedDescription': 'string',
			# 'user_join_date': 'datetime64[ns]',
			# 'user_followers': 'int64',
			# 'user_location': 'string',
			'user_verified': 'boolean',
			# 'inReplyToTweetId': 'string'
		}

		if progressbar:
			progressbar.progress(i / (max_tweets + 1))

		if i > max_tweets:
			break
	
	return pd.DataFrame.from_dict(tweets_list, orient='index').astype(dtypes)

def sentiment_classification(series_content, progressbar):
	"""Classifies the sentiment of a text using pysentimiento module."""

	analyzer = create_analyzer(task="sentiment", lang="es")

	output = pd.Series(index=series_content.index)

	pos = pd.Series(index=series_content.index)
	neu = pd.Series(index=series_content.index)
	neg = pd.Series(index=series_content.index)

	for i, content in enumerate(series_content):
		output[i] = analyzer.predict(content).output
		pos[i] = round(analyzer.predict(content).probas['POS'], 4)
		neu[i] = round(analyzer.predict(content).probas['NEU'], 4)
		neg[i] = round(analyzer.predict(content).probas['NEG'], 4)

		if progressbar:
			progressbar.progress(i / (len(series_content) + 1))

	return output, pos, neu, neg

def preprocess(series_content, progressbar):
	"""Preprocesses the text of a tweet."""

	result = pd.Series(index=series_content.index)

	for i, content in enumerate(series_content):
		result[i] = preprocess_tweet(content)

		if progressbar:
			progressbar.progress(i / (len(series_content) + 1))

	return result

def emotion_classification(series_content, progressbar):
	"""Classifies the emotion of a text using pysentimiento module."""

	analyzer = create_analyzer(task="emotion", lang="es") # hate_speech

	output = pd.Series(index=series_content.index)
	others = pd.Series(index=series_content.index)
	joy = pd.Series(index=series_content.index)
	sadness = pd.Series(index=series_content.index)
	anger = pd.Series(index=series_content.index)
	surprise = pd.Series(index=series_content.index)
	disgust = pd.Series(index=series_content.index)
	fear = pd.Series(index=series_content.index)

	for i, content in enumerate(series_content):
		output[i] = analyzer.predict(content).output
		others[i] = analyzer.predict(content).probas['others']
		joy[i] = analyzer.predict(content).probas['joy']
		sadness[i] = analyzer.predict(content).probas['sadness']
		anger[i] = analyzer.predict(content).probas['anger']
		surprise[i] = analyzer.predict(content).probas['surprise']
		disgust[i] = analyzer.predict(content).probas['disgust']
		fear[i] = analyzer.predict(content).probas['fear']

		if progressbar:
			progressbar.progress(i / (len(series_content) + 1))

	return output, others, joy, sadness, anger, surprise, disgust, fear

def convert_df(df):
   """Convert a pandas dataframe into a csv file that can be downloaded"""
   return df.to_csv(index=False).encode('utf-8')
