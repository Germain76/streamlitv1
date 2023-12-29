###############################################################################
#                                  FUNCTIONS                                  #
###############################################################################

import streamlit as st
import src.backend as backend
from src.sidebar import sidebar
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.express import bar, pie


def next_image():
    if	st.session_state.img_id == 9:
        st.session_state.img_id = 0
    else:
        st.session_state.img_id+=1

def prev_image():
    if	st.session_state.img_id == 0:
        st.session_state.img_id = 9
    else:
        st.session_state.img_id-=1

def load_new_images():
    st.session_state.img_id = 0
    st.session_state.img_exploration_data = backend.fetch_unprocessed_images(backend.df_to_raw_images(backend.df.sample(n=10, random_state=np.random.randint(0, 1))).to_list())

###############################################################################
#                                 GRAPHIQUES                                  #
###############################################################################

@st.cache_data
def graph1():
    colors = ["Small" for i in range(6)]
    colors+=["Med" for i in range(20)]
    colors+=["Big"]
    annotations = ["Cette variable est sous-représentée" for i in range(6)]
    annotations += ["L'effectif de cetre variable est dans la norme" for i in range(20)]
    annotations += ["Cette variable est sur-représentée"]
    plotly_df = backend.df["prdtypecode"].value_counts(ascending=True).reset_index()
    plotly_df["colors"] = colors
    plotly_df["effectif"] = annotations
    fig = bar(plotly_df, x="prdtypecode", y="count", color="colors", color_discrete_map={"Med":"#123c60",  "Big":"#f0ca9d",  "Small":"#ef3b6e"}, hover_name="prdtypecode", hover_data={"count":True, "prdtypecode":False, "effectif":False, "colors":False})
    fig.update_layout(xaxis_type="category", showlegend=False, xaxis_tickangle=-45, title_text="Répartition des classes")
    fig.update_traces(hovertemplate='<b>%{hovertext}</b><br>Effectif : %{y}<br>%{customdata[0]}<extra></extra>')
    fig.update_xaxes(title_text="Codes produits")
    return fig

@st.cache_data
def graph2(description):
    description_counts = pd.DataFrame(description.isna().value_counts())
    description_counts["name"] = ["Avec description", "Sans description"]
    fig = pie(description_counts, values="count", title="Pourcentage de produits avec et sans description", hole=0.5, hover_name="name", color="name", color_discrete_map={"Avec description":"#123c60",  "Sans description":"#ef3b6e"}, names=["Avec", "Sans"])
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def chart1():
    sns.set(style="whitegrid")
    df_sorted = df_no_processing["prdtypecode"].value_counts().reset_index()
    df_sorted.columns = ["prdtypecode", "count"]
    plt.figure(figsize=(7, 5))
    palette = sns.color_palette("Reds_r", len(df_sorted))
    sns.barplot(
        x="prdtypecode",
        y="count",
        data=df_sorted,
        order=df_sorted["prdtypecode"],
        palette=palette,
    )
    plt.title("Distribution des types de produits")
    plt.xlabel("Code de type de produit")
    plt.ylabel("Nombre de produits")
    plt.xticks(rotation=90)
    st.pyplot(plt)

def chart2():
    df_no_processing["description_length"] = df_no_processing["description"].apply(
        lambda x: len(x) if pd.notna(x) else 0
    )
    groupe_par_prdtype_description = (
        df_no_processing.groupby("prdtypecode")["description_length"].mean().reset_index()
    )
    palette = sns.color_palette(
        "magma", n_colors=len(groupe_par_prdtype_description)
    )
    plt.figure(figsize=(7, 5))
    sns.barplot(
        x="prdtypecode",
        y="description_length",
        data=groupe_par_prdtype_description,
        palette=palette,
    )
    plt.xlabel("Code de produit (prdtypecode)")
    plt.xticks(rotation=90)
    plt.ylabel("Taille moyenne des descriptions")
    plt.title("Taille moyenne des descriptions par prdtypecode")
    st.pyplot(plt)


def chart3():
    df_no_processing["designation_length"] = [len(df_no_processing.loc[i, "designation"]) for i in range(len(df_no_processing))]
    groupe_par_prdtype_designation = (
        df_no_processing.groupby("prdtypecode")["designation_length"].mean().reset_index()
    )
    palette = sns.color_palette(
        "magma", n_colors=len(groupe_par_prdtype_designation)
    )
    plt.figure(figsize=(7, 5))
    sns.barplot(
        x="prdtypecode",
        y="designation_length",
        data=groupe_par_prdtype_designation,
        palette=palette,
    )
    plt.xlabel("Code de produit (prdtypecode)")
    plt.xticks(rotation=90)
    plt.ylabel("Taille moyenne des designation")
    plt.title("Taille moyenne des designation (titre) par prdtypecode")
    st.pyplot(plt)


def chart4():
    df_no_processing["a_description"] = df_no_processing["description"].isna()
    description_counts = df_no_processing["a_description"].value_counts(normalize=True)
    plt.figure(figsize=(8, 8))
    colors = ["#8B0000", "#F08080"]
    # S'assurer que 'Avec Description' est en premier
    description_counts = description_counts.reindex([True, False])
    plt.pie(
        description_counts,
        labels=["Avec Description", "Sans Description"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
    )
    plt.title("Pourcentage de produits avec et sans description")
    st.pyplot(plt)


###############################################################################
#                            VARIABLES DECLARATIONS                           #
###############################################################################


if 'img_id' not in st.session_state:
    st.session_state.img_id = 0
if "img_exploration_data" not in st.session_state:

    st.session_state.img_exploration_data = backend.fetch_unprocessed_images(backend.df_to_raw_images(backend.df1.sample(n=10)).to_list())
df_no_processing = backend.load_df_no_processing()

###############################################################################
#                             SCRIPT INSTRUCTIONS                             #
###############################################################################

sidebar()
st.title("Exploration des données")

text_exploration, images_exploration = st.tabs(["Exploration du texte", "Exploration des images"])

with text_exploration:
    st.subheader("Exploration du texte")
    st.subheader("*Jeu de données*")
    st.write(f"Notre dataset contient **{len(backend.df)}** lignes. Voici les premières lignes.")
    st.dataframe(backend.load_raw_df().head())
    st.write("Le dataframe contient quatres colonnes :")
    st.markdown("""
             - **designation** : Libellé du produit
             - **description** : Description du produit
             - **productid** : Identifiant unique du produit
             - **imageid** : ID de l'image correspondante
    """)

    st.subheader("*Variable cible*")
    st.write("**prdtypecode** : La variable cible est une variable catégorielle représentant le code produit de chaque observation. Le code est un entier positif et la variable a 27 modalités..")
    st.dataframe([[backend.load_y_df().prdtypecode.unique()]], column_config={"0": st.column_config.ListColumn("Valeurs uniques de la variable cible")})


    # Affichage du graphique en fonction du choix
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distribution", "Quantité de texte par classe", "Valeur manquantes ", "Boxplot", "Exploration des langues du Dataframe"])
    with tab1:
        st.plotly_chart(graph1())
    with tab2:
        chart2()
        chart3()
    with tab3:
        st.plotly_chart(graph2(df_no_processing["description"]))
    with tab4:
        st.image(
            "./images/statistiques_descriptions_designations.png",
        )
        st.markdown(
            """
                Statistiques descriptives de la longueur des descriptions :
                - count    84916.000000
                - mean       524.555926
                - std        754.893905
                - min          0.000000
                - 25%          0.000000
                - 50%        231.000000
                - 75%        823.000000
                - max      12451.000000

                Statistiques descriptives de la longueur des designations :
                - count    84916.000000
                - mean        70.163303
                - std         36.793383
                - min         11.000000
                - 25%         43.000000
                - 50%         64.000000
                - 75%         90.000000
                - max        250.000000
            """
        )
    with tab5:
        st.subheader("Pipeline")
        pipeline = """
            x_train = pd.read_csv(path + "X_train_update.csv", sep=",", index_col=0)
            y_train = pd.read_csv(path + "Y_train_CVw08PX.csv", sep=",", index_col=0)
            df = x_train.join(y_train)
            df["description"].fillna("", inplace=True)
            df = df.dropna()
            df.description = df.description.apply(functions.deleteHTML)
            df.designation = df.designation.apply(functions.deleteHTML)
            df = df.dropna()
            df["langage_description"] = df["description"].apply(functions.ignore_translate_text)
            df["langage_designation"] = df["designation"].apply(detect)
            ##Suppression des index où la détection n'a pas pu se faire, suppression  de 73 index
            df = df.dropna()
            correspondance = df.langage_designation== df.langage_description
            taux_correspondance = correspondance.mean() * 100
            """
        st.code(pipeline, language='python')
        st.image(
            "./images/langues.png",
        )
        st.markdown(
            """
            Les langues présentes dans le jeu de données sont : 
            ['fr', 'en', 'it', 'ca', 'pt', 'pl', 'de', 'ro', 'id', 'vi','tl', 'es', 'nl', 'fi', 'hr', 'et', 'sk', 'sl', 'sv', 'no', 'af','da', 'sw', 'so', 'cy', 'cs', 'tr', 'hu', 'lt', 'lv']
            """
        )
        st.markdown(
            """
            Le taux de correspondance entre les 2 colonnes langage_description et langage_designation est de 70%
            """
        )


with images_exploration:
	st.subheader("Exploration des images")
	st.write(f"Notre dataset contient **{len(backend.df)}** images. En voici quelques unes.")
	st.markdown("")
	st.markdown("""<p>10 images du jeu de données</p>""",unsafe_allow_html=True)
	with st.container(border=True):
		
		left_col, right_col = st.columns([5, 2.5])
		with left_col:
			with st.container(border=True):
				st.image(st.session_state.img_exploration_data[st.session_state.img_id], channels="BGR")
		with right_col:
			with st.container(border=True):
				st.markdown(f"""<p style="text-align: center;"><strong>{st.session_state.img_id+1}/10</strong></p>""", unsafe_allow_html=True)
			c1, c2 = st.columns([1.3,1])
			with c1:
				st.button("Précédente", on_click=prev_image)
			with c2:
				st.button("Suivante", on_click=next_image)
			st.button("&ensp; Charger d'autres images&emsp;", on_click=load_new_images)

	st.write("La majorité des images ont des bordures blanches autour du sujet. Sur certaines images, ces bordures occupent la majorité de l'espace.")
	st.write("Pour améliorer les performances du modèle, nous allons essayer de réduire la quantité de ces pixels vides.")