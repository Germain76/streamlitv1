###############################################################################
#                                  FUNCTIONS                                  #
###############################################################################

import streamlit as st
from src.sidebar import sidebar
from pandas import DataFrame
import src.backend as backend
from numpy.random import randint
from pandas import DataFrame

def set_random_line():
	st.session_state.selected_line = randint(0, len(backend.df))

###############################################################################
#                            VARIABLES DECLARATIONS                           #
###############################################################################

if 'is_multimodal_loaded' not in st.session_state:
    st.session_state.is_multimodal_loaded = False
if 'selected_line' not in st.session_state:
    st.session_state.selected_line = 46942

###############################################################################
#                             SCRIPT INSTRUCTIONS                             #
###############################################################################

# Title
sidebar()
st.title("Classifieur multimodal")

tab1, tab2 = st.tabs(["Description du modèle", "Démo"])
with tab1:
	# Architecture
	st.subheader("Architecture")
	st.markdown("Notre modèle repose sur le **CNN basé sur EfficientNetV2 B3** pour traiter les images, et un modèle **LSTM** pour traiter le texte. Chacun des modèles effectue **indépendamment une prédiction**, et retourne une matrice de probabilités. Les probabilités renvoyées par les modèles vont chacune passer par une couche de normalisation (Batch Norm layer) indépendante, puis être concaténées ensembles. Elles sont ensuite traités par une couche dense de 512 neurones.")
	st.write("Voici le schéma du modèle final :")
	_, col = st.columns([1,8])
	with col:
		st.image("images/Multimodal.png")

	# Result
	st.subheader("Résultats")
	st.markdown("""<h5>Ensemble de test</h5>""", unsafe_allow_html=True)
	with st.container(border=True):
		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("Accuracy", 0.95)
		with col2:
			st.metric("Recall", 0.96)
		with col3:
			st.metric("F1-score", 0.96)

	st.write("Le modèle final obtient d'excellents résultats, supérieur à ceux des deux modèles primitifs.")

with tab2:
	# Live testing
	st.subheader("Démo")
	st.write("Ci-dessous un widget permettant de tester le classifieur multimodal, et de voir en action toute notre pipeline de traitement des données.")
	# Widget
	with st.container(border=True):
		# Load model button
		if not st.session_state.is_multimodal_loaded:
			if st.button("Charger le modèle") :
				with st.spinner("Chargement du modèle..."):
					multimodal_model = backend.load_multimodal_classifier()
					st.session_state.is_multimodal_loaded = True
					st.rerun()
		# If model is loaded
		else:
			# Choose line	
			col1, col2 = st.columns([4,1])
			with col1:
				st.session_state.selected_line = st.number_input("Ligne", min_value=0, max_value=len(backend.df), value=st.session_state.selected_line)
			with col2:
				st.markdown("""<div style="margin-bottom:3px">&emsp;</div>""",unsafe_allow_html=True)
				st.button("Aléatoire", on_click=set_random_line)
			st.divider()

			# Display raw data
			st.markdown("""<h4>Données brutes</h4>""",unsafe_allow_html=True)
			line = backend.load_raw_df().loc[st.session_state.selected_line, :] # Get line froom raw dataframe
			line = line.fillna(value="")
			image_name = "image_" + str(line["imageid"])+ "_product_" + str(line["productid"]) + ".jpg"
			col1, col2 = st.columns(2)
			with col1:
				# Display images
				st.markdown("""<p>Image</p>""",unsafe_allow_html=True)
				with st.container(border=True):		
					st.image(backend.fetch_unprocessed_images([image_name]), channels="BGR")
			with col2:
				# Display text data : title and description
				st.markdown("""<p>Texte</p>""",unsafe_allow_html=True)
				with st.container(border=True):
					with st.expander(f"Longeur totale : {len(str(line['description']+line['designation']))} caractères"):
						st.markdown("**Titre :** "+str(line['designation']))
						st.markdown("**Description :** "+str(line['description']))
			st.divider()

			# Display preprocessed data
			line = backend.df.loc[st.session_state.selected_line, :]
			image_name = str(line["imageid"])+ "_product_" + str(line["productid"]) + ".jpg"
			vectorized_text = backend.vectorize_text([line["description"]])
			preprocessed_image = backend.fetch_processed_images([str(line["prdtypecode"])+"/"+image_name])[0]
			st.markdown("""<h4>Données traitées</h4>""",unsafe_allow_html=True)
			col1, col2 = st.columns(2)
			with col1:
				st.markdown("""<p>Image</p>""",unsafe_allow_html=True)
				with st.container(border=True):		
					st.image(preprocessed_image, channels="BGR")
			with col2:
				st.markdown(f"""<p>Texte</p>""",unsafe_allow_html=True)
				with st.container(border=True):
					with st.expander(f"Longeur totale : {len(str(line['description']))} caractères"):
						st.markdown("**Texte nettoyé :** "+line["description"])
						st.markdown("**Texte vectorisé :** ")
						st.table(vectorized_text)
			st.divider()

			# Display model prediction
			st.markdown("""<h4>Résultat final</h4>""",unsafe_allow_html=True)
			pred = backend.predict_with_multimodal(preprocessed_image, backend.pad_text(vectorized_text))
			with st.container(border=True):
				col1, col2, col3 = st.columns([1,1,1.2])
				with col1:
					st.metric("Prédiction", pred)
				with col2:
					st.metric("Classe réelle", line["prdtypecode"])
				with col3:
					st.markdown("""<p style="height:0px"></p>""",unsafe_allow_html=True)
					if pred == line["prdtypecode"]:
						st.success("&ensp;Prédiction correcte ✔")
					else:
						st.error("&ensp;Prédiction incorrecte ❌")
