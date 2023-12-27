###############################################################################
#                                  FUNCTIONS                                  #
###############################################################################

import streamlit as st
from src.sidebar import sidebar

###############################################################################
#                             SCRIPT INSTRUCTIONS                             #
###############################################################################

sidebar(auto_expand=True)
st.title("Conclusion")
conclusion, outils, remerciements = st.tabs(["Conclusion", "Outils utilisés","Remerciements"])

with conclusion:
	st.markdown("Le modèle final rempli les objectifs attendus. En se basant sur une **image d'un produit** ainsi qu'une **description textuelle**, le modèle arrive à renvoyer une **prédiction de la classe produit.**")
	st.markdown("Il classifie correctement **95%** des produits, ce qui en fait un excellent modèle.") 
	st.markdown("Au moment de la rédaction de cette conclusion nous n'avons pas eu le temps de soumettre le modèle sur le site du challenge, mais nous avons de bonnes raisons de croire que nous pourrons remplir notre objectif de debut : **dépasser ou égaler FEEEscientest**.")
	st.markdown("La pipeline d'ingestion et de traitement des données est efficace, et peut être automatisée à l'aide d'outils comme un `tf.data.Dataset`. Il serait donc possible de **déployer ce modèle dans un cloud**,  puis d'utiliser la pipeline pour traiter de nouvelles données et réaliser un entrainement continu.")
	
	st.header("Pistes d'améliorations")
	st.markdown("Durant la réalisation de ce modèle, nous avons rencontrés des difficultés lors de la traduction du texte, ainsi qu'à cause de la qualité variable des images. Il serait donc probablement possible d'améliorer les résultats en :")
	st.markdown("""
		- **... utilisant d'autres bibliothèques de traduction :** Nous nous sommes limitées aux bibliothèques gratuites, mais il est probable que certaines bibliothèques payantes soient capables d'offrir une traduction plus consistente que celles que nous utilisons acutellement.
		- **... normalisant les images :** Les images sont d'une qualité fortement variable. Certaines sont bien cadrées et bien exposées, tandis que d'autres sont floues, mal cadrées ou trop sombres. La mise en place d'un système pour standardiser les images associées à chaque produit pourrait contribuer à améliorer les performances.
                """)
	st.header("Le mot de la fin")
	st.markdown("En résumé, ce projet a été une belle occasion pour notre équipe d'acquérir une **solide expérience professionnelle** dans le domaine de **l'intelligence artificielle**.") 
	st.markdown("En plongeant dans les défis complexes de ce domaine, nous avons développé des **compétences techniques** pointues et affiné notre compréhension des nuances liées à l'intégration de l'IA dans des projets concrets.") 
	st.markdown("Au-delà des résultats obtenus, cette expérience a été une véritable **source de croissance professionnelle pour chacun d'entre nous**, renforçant notre capacité à relever avec confiance les défis futurs de l'intelligence artificielle.")
	st.markdown("")
	st.markdown("")
	st.markdown("**Merci de votre attention !**") 
	st.markdown("")
	st.markdown("*Nous espérons que vous aurez trouvé cette présentation intéressante, et qu'elle vous aura donné envie d'en apprendre plus sur les modèles multimodaux.*")
	st.divider()
	st.markdown("""<p style="text-align:center"> &#129302;</p>""", unsafe_allow_html=True)


with outils:
	st.write("Outils utilisés pour la réalisation du projet")
	st.markdown("""
		- OpenCV
		- Keras
		- Tensorflow
		- Pandas
		- Numpy
		- Scikit-Learn
		- NLTK
		- MTranslate
		- Langdetect
		- Spellcheck
		- BeautifulSoup
	""")

	st.write("Outils utilisés pour la présentation du projet")
	st.markdown("""
		- Streamlit
		- Plotly
		- Seaborn
		- Matplotlib
		- Figma
	""")

with remerciements:
	with st.container(border=True):
		st.markdown("Merci à **Maxime MICHEL**, notre mentor de projet pour ses conseils durant la réalisation de ce projet.")
