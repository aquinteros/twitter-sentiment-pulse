from resources.functions import *

def run():

	st.title("Twitter Sentiment Pulse")

	st.write("Esta aplicación permite obtener tweets desde la API de Twitter usando el módulo snscrape, también permite preprocesar y clasificar los tweets usando un modelo de análisis de sentimiento llamado pysentimiento, construido en el framework huggingface.")

	buy_me_a_coffee(username="aquinteros", floating=False, width=221)

	tweet_form = st.form("tweet_form")

	concept = tweet_form.text_input("Concepto")

	c1, c2, c3 = tweet_form.columns(3)

	start_date = c1.date_input("Fecha Inicio")

	end_date = c2.date_input("Fecha Fin")

	max_tweets = c3.number_input("Max tweets", min_value=100, max_value=100_000, value=100, step=100)

	if tweet_form.form_submit_button("Obtener Tweets"):

		if concept == "":
			st.error("Llena el campo Concepto")
			return
		
		st.write("Scraping tweets...")
		
		progressbar = st.progress(0)

		tweets_list = get_tweets(concept, str(start_date), str(end_date), progressbar, max_tweets=max_tweets)

		tweets_list['date'] = pd.to_datetime(tweets_list['datetime']).dt.date

		if tweets_list.empty:
			st.error("No se encontraron tweets")
			return
		
		st.session_state.tweets_list = tweets_list

		progressbar.progress(100)

		st.write("Preprocesando...")

		progressbar = st.progress(0)

		tweets_list['preprocessed'] = preprocess(tweets_list['rawContent'], progressbar)

		progressbar.progress(100)

	if 'tweets_list' in st.session_state:

		st.write(f"Cantidad de tweets importados: {len(st.session_state.tweets_list)}")

		if st.button("Clasificar Polaridad"):
			
			if 'tweets_list' not in st.session_state:
				st.error("No se encontraron tweets")
				return
			
			tweets_list = st.session_state.tweets_list

			st.write("Calculando Polaridad...")

			progressbar = st.progress(0)

			tweets_list['sentiment'], tweets_list['POS'], tweets_list['NEU'], tweets_list['NEG'] = sentiment_classification(tweets_list['preprocessed'], progressbar)

			progressbar.progress(100)

			st.success("Listo!")

			grouped = tweets_list.groupby(['date', 'sentiment']).size().reset_index(name='counts')

			fig = px.bar(grouped, x='date', y='counts', color='sentiment', title='Sentiment', width=800, height=500)

			st.plotly_chart(fig)

		if st.button("Clasificar Emoción"):

			if 'tweets_list' not in st.session_state:
				st.error("No se encontraron tweets")
				return
			
			tweets_list = st.session_state.tweets_list
			
			st.write("Calculando Emociones...")

			progressbar = st.progress(0)

			tweets_list['emotion'], tweets_list['others'], tweets_list['joy'], tweets_list['sadness'], tweets_list['anger'], tweets_list['surprise'], tweets_list['disgust'], tweets_list['fear']  = emotion_classification(tweets_list['preprocessed'], progressbar)

			progressbar.progress(100)

			st.success("Listo!")

			# grouped = tweets_list.groupby(['date', 'output']).size().reset_index(name='counts')

			# fig = px.bar(grouped, x='date', y='counts', color='output', title='Sentiment', width=800, height=500)

			# st.plotly_chart(fig)
		
		st.dataframe(tweets_list, use_container_width=True)

		csv = tweets_list.to_csv(index=False)

		st.download_button('Descarga Tweets en csv', csv, "tweets_" + concept + ".csv", "text/csv", key='download-csv')

		st.download_button('Descarga Tweets en parquet', tweets_list.to_parquet(), "tweets_" + concept + ".parquet", "application/octet-stream", key='download-parquet')
	
	st.markdown("""
		¿Tienes preguntas? \n
		Envíame un correo: \n
		"""
	)

	mention(
		label="alvaro.quinteros.a@gmail.com",
		icon="📧",
		url="mailto:alvaro.quinteros.a@gmail.com"
	)

	st.markdown("""
		O abre un issue en el repositorio de github: \n
		"""
	)  

	mention(
		label="twitter-sentiment-pulse",
		icon="github",
		url="https://github.com/aquinteros/twitter-sentiment-pulse"
	)
	
	with st.expander("Referencia pysentimiento"):
		st.markdown("""
		```
		@misc{perez2021pysentimiento,
			title={pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks},
			author={Juan Manuel Pérez and Juan Carlos Giudici and Franco Luque},
			year={2021},
			eprint={2106.09462},
			archivePrefix={arXiv},
			primaryClass={cs.CL}
		}
		```
		""")

if __name__ == "__main__":
	run()