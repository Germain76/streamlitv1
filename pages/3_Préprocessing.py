###############################################################################
#                                  FUNCTIONS                                  #
###############################################################################

import streamlit as st
import src.backend as backend
from src.sidebar import sidebar
from numpy import uint8
from cv2 import cvtColor, threshold, findContours, resize, boundingRect, rectangle,contourArea, COLOR_BGR2GRAY, THRESH_BINARY_INV, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
from langdetect import detect
import time
import src.functions as functions

@st.cache_resource
def run_pipeline(df):
    with st.status("Traitement du texte...", expanded=True, state="running"):
        df1 = df.iloc[0:13,]
        df1["description"].fillna("", inplace=True)
        st.write("(1/9) Suppression des balises HTML...")
        df1.description = df1.description.apply(functions.deleteHTML)
        df1.designation = df1.designation.apply(functions.deleteHTML)

        st.write("(2/9) Fusion des titres et des descriptions...")
        df1["identification"] = (
            df1["designation"].astype(str) + " . " + df1["description"].astype(str)
        )

        st.write("(3/9) Detection des langues...")
        df1 = df1.drop(["designation", "description"], axis=1)
        df1["langage_identification_avant_traduction"] = df1["identification"].apply(
            detect
        )
        st.write("(4/9) Suppression des emojis...")
        df1["identification"] = df1["identification"].apply(functions.delete_emoji)

        st.write("(5/9) Correction des textes...")
        df1["identification"] = df1.apply(
            lambda row: functions.correct_text(
                row["identification"], row["langage_identification_avant_traduction"]
            ),
            axis=1,
        )
        st.write("(6/9) Traduction...")
        df1["identification"] = df1.apply(
            lambda row: functions.traduire_vers_françaisv3(
                row["identification"], row["langage_identification_avant_traduction"]
            ),
            axis=1,
        )
        df1["langage_identification_apres_traduction"] = df1["identification"].apply(
            detect
        )

        st.write("(7/9) Suppression de la ponctuation...")
        df1["identification"] = functions.delete_punctuation(df1["identification"])
        df2 = df1.iloc[0:13,]

        st.write("(8/9) Suppression des mots vides...")
        df1["identification"] = df1["identification"].apply(functions.remove_stopwords)
        df1["identification"] = df1["identification"].str.lower()

        st.write("(9/9) Tokenisation...")
        df1["identification"] = df1["identification"].apply(functions.tokenize_text)
        st.write("Succès !")
    st.write("Pipeline terminée.")
    return df1, df2

@st.cache_data
def load_preprocess_images_page():
	img_gray = cvtColor(original_image, COLOR_BGR2GRAY)
	_, img_contrasted = threshold(img_gray, thresh=240, maxval=255, type=THRESH_BINARY_INV)
	contours, _ = findContours(img_contrasted, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
	contours = max(contours, key=contourArea)
	coords = boundingRect(contours)
	x, y, w, h = coords
	x, y, w, h = (x, y, w, h) if w == h else (y, y, h, h) if w < h else (x, x, w, w) # Preserve apect ratio black magic
	processed_image = resize(original_image[y : y + h, x : x + w], (500,500))
	return img_contrasted, contours, coords, x, y, w, h, processed_image


###############################################################################
#                            VARIABLES DECLARATIONS                           #
###############################################################################


if 'is_pipeline_loaded' not in st.session_state:
    st.session_state.is_pipeline_loaded = False

original_image = original_image=uint8((backend.fetch_unprocessed_images(["image_482917_product_928735.jpg"])[0])*255)
img_contrasted, contours, coords, x, y, w, h, processed_image = load_preprocess_images_page()
pipeline = """

df["description"].fillna("", inplace=True)
df.description = df.description.apply(functions.deleteHTML)
df.designation = df.designation.apply(functions.deleteHTML)
df["identification"] = (
df["designation"].astype(str) + " . " + df["description"].astype(str))
df = df.drop(["designation", "description"], axis=1)
df['langage_identification_avant_traduction'] = df['identification'].apply(detect)
df["identification"] = df["identification"].apply(functions.delete_emoji)
df['identification'] = df.apply(lambda row: functions.correct_text(row['identification'], row['langage_identification_avant_traduction']), axis=1)
df["identification"] = df.apply(lambda row: functions.traduire_vers_françaisv3(row["identification"], row['langage_identification_avant_traduction']), axis=1)
df['langage_identification_apres_traduction'] = df['identification'].apply(detect)
df['identification'] = functions.delete_punctuation(df['identification'])
df['identification'] = df['identification'].apply(functions.remove_stopwords)
df['identification'] = df['identification'].str.lower()
df['identification'] = df['identification'].apply(functions.tokenize_text)
df.to_csv('dataprepreprocessing.csv',index=False)
"""

df = backend.load_df_no_processing()

###############################################################################
#                             SCRIPT INSTRUCTIONS                             #
###############################################################################

# Title
sidebar()
st.title("Preprocessing")
images_preprocessing, text_preprocessing  = st.tabs(["Traitement des images", "Traitement du texte"])

# Text
with text_preprocessing:
	st.write("DataFrame Original:")
	st.dataframe(df.head(13))
	st.title("Pipeline")
	# affichage du code de la pipeline
	st.code(pipeline, language="python")
	# bouton pour afficher la pipeline
	if not st.session_state.is_pipeline_loaded:
		if st.button("Lancer la Pipeline"):
			df1, df2 = run_pipeline(df)
			st.session_state.is_pipeline_loaded = True
			st.rerun()
	else:
		# afficher le Dataframe préprocessé
		st.write("Dataframe après le préprocessing")
		df1, df2 = run_pipeline(df)
		st.dataframe(df1.head(13))
		st.dataframe(df2.head(13))


# Images
with images_preprocessing:
	st.subheader("Agrandissement du sujet")
	st.markdown("La première étape du preprocessing des images consiste à **maximiser la taille du sujet**.")

	# Widget
	st.markdown("Prenons la ligne **84 312** comme exemple. La majorité de l'image est occupée par des **bordures blanches**.")
	st.dataframe(backend.df[backend.df["imageid"] == 482917])
	col, _ = st.columns([2, 2])
	with col:
		with st.container(border=True):
			st.image(original_image, channels="BGR")
	with st.expander("Traitement des images pour l'agrandissement du sujet"):
		st.write("L'agrandissement du sujet déroule en 3 étapes.")
		tab1, tab2, tab3, tab4 = st.tabs(["Thresholding", "Bounding box", "Recadrement", "Résultat"])
		with tab1:
			st.subheader("Etape 1 : Thresholding")
			st.write("Après avoir converti l'image en niveau de gris, nous appliquons un thresholding pour séparer les pixels blancs en bordure de l'image de tous les autres pixels.")
			col, _ = st.columns([4, 1])
			with col:
				with st.container(border=True):
					st.image(img_contrasted)

		with tab2:
			st.subheader("Etape 2 : Bounding box")
			st.write("Nous allons ensuite extraire le sujet en recherchant les contours extérieurs présents au sein de l'image. On peut ensuite récuperer le rectangle (bounding box) englobant ces contours.")
			col, _ = st.columns([4, 1])
			with col:
				with st.container(border=True):
					st.image(rectangle(original_image,(coords[0],coords[1]),(coords[0]+coords[2],coords[1]+coords[3]),(255,255,0),2), channels="BGR")

		with tab3:
			st.subheader("Etape 3 : Recadrement")
			st.markdown("""Une fois que l'on a extrait la bounding box contenant tous les pixels utiles de l'image, on peut recadrer l'image selon la bounding box. <br> :warning: La reconnaissance d'image est sensible aux proportions des objets. Il faut donc vérifier la forme de la bounding box avant de recardrer l'image. Si la bounding box n'est pas un carré, le recadrement de l'image risque d'entrainer un étirement de l'image qui diminuerait les performances du modèle. <br> On transforme donc la bounding box en carré avant d'effectuer le recardement""", unsafe_allow_html=True)
			col, _ = st.columns([4, 1])
			with col:
				with st.container(border=True):
					st.image(rectangle(original_image,(x,y),(x+w,y+h),(255,0,0),2), channels="BGR")
			
		with tab4:
			st.subheader("Etape 3 : Résultat")
			col, _ = st.columns([4, 1])
			with col:
				with st.container(border=True):
					st.image(processed_image, channels="BGR")

	st.subheader("Equilibrage des classes")
	st.write("La classe 2583 est largement majoritaire dans notre dataset. On applique donc un undersampling et retirons 2000 images aléatoires de cette classe du jeu de données. On calcule également des poids de classes inversements proportionnels à leur effectif pour aider le modèle à classifier correctement les classes minoritaires.")
