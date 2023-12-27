import pandas as pd
import numpy as np
import nltk
import spacy
from textwrap import shorten
import re
from langdetect import detect, LangDetectException
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from spellchecker import SpellChecker
import spacy
from googletrans import Translator
from mtranslate import translate

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
#from deep_translator import GoogleTranslator
import string


def tokenize_text(text):
    return word_tokenize(text)


##########################################################################################################
# Définissez une fonction pour supprimer les caractères spéciaux
def remove_special_characters(text):
    # Utilisez une expression régulière pour supprimer les caractères spéciaux
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text


##########################################################################################################
# Définissez une fonction pour supprimer les mots vides
def remove_stopwords(text):
    stop_words = set(stopwords.words("french"))
    words = text.split()  # Divisez le texte en mots
    filtered_words = [
        word for word in words if word.lower() not in stop_words
    ]  # Supprimez les mots vides
    filtered_text = " ".join(filtered_words)  # Rejoignez les mots restants en un texte
    return filtered_text


##########################################################################################################
# Fonction qui supprime les balises HTML
def delete_HTMLv0(text):
    re.sub(r"<(?!img|b|bold|strong|a|iframe|object).*?>", "", str(text))
    return text


##########################################################################################################
# Fonction qui supprime les balises HTML v2
def deleteHTML(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    return text


##########################################################################################################
# Fonction qui supprime la ponctuation
def delete_punctuation(text):
    text = re.sub(r"[^\w\s]", "", str(text))
    return text


##########################################################################################################
# Si la traduction ne peut pas se faire, langdetect passe à l'index suivant du dataframe
def ignore_translate_text(text):
    try:
        return detect(text)
    except LangDetectException:
        return None


##########################################################################################################
# Correction texte en anglais
def correct_en(text):
    spell_en = SpellChecker()
    corrected_text = []
    for word in text.split():
        corrected_word = (
            spell_en.correction(word) if spell_en.correction(word) is not None else word
        )
        corrected_text.append(corrected_word)
    return " ".join(corrected_text)


##########################################################################################################
def traduire_vers_françaisv3(text, lg):
    translator = Translator()
    if lg == "en" or lg == "de":
        text = translator.translate(text, src=lg, dest="fr").text
    return text


################################################################################
# def traduire_vers_francaisv4(texte):
#     try:
#         # Si le texte est déjà en français, ne le traduisez pas
#         if detect(texte) == "fr":
#             return texte
#         else:
#             # Traduire le texte vers le français
#             translator = GoogleTranslator(source="en", target="fr")
#             translation = translator.translate(texte)
#             return translation
#     except:
#         # En cas d'erreur, renvoyer le texte d'origine
#         return texte


##########################################################################################################
# Correction texte en français
def correct_fr(text):
    spell_en = SpellChecker(language="fr")
    corrected_text = []
    for word in text.split():
        corrected_word = (
            spell_en.correction(word) if spell_en.correction(word) is not None else word
        )
        corrected_text.append(corrected_word)
    return " ".join(corrected_text)


##########################################################################################################
# Correction texte en allemand
def correct_de(text):
    spell_en = SpellChecker(language="de")
    corrected_text = []
    for word in text.split():
        corrected_word = (
            spell_en.correction(word) if spell_en.correction(word) is not None else word
        )
        corrected_text.append(corrected_word)
    return " ".join(corrected_text)


##########################################################################################################
def correct_text(text, language):
    if language == "en":
        return correct_en(text)
    elif language == "fr":
        return correct_fr(text)
    elif language == "de":
        return correct_de(text)
    else:
        return text


##########################################################################################################
def traduire_vers_francaisv1(texte, lg):
    if lg != "fr":
        translator = Translator()
        try:
            traduction = translator.translate(texte, dest="fr")
            return traduction
        except:
            return texte
    else:
        return texte


##########################################################################################################
def traduire_vers_francais(texte):
    translator = Translator()
    try:
        traduction = translator.translate(texte, dest="fr").text
        return traduction
    except:
        return None


##########################################################################################################
def traduire_vers_francaisv2(texte):
    translator = Translator()
    try:
        traduction = translator.translate(texte, dest="fr").text
        return traduction
    except:
        return None


##########################################################################################################
# Suppression des URL
def delete_url(text):
    text = re.sub(r"http\S+", "", str(text))
    return text


##########################################################################################################
# Suppression de la ponctuations
def delete_punctuation(text):
    # Liste de ponctuations
    punctuations = r"[-@#!?+&*[\]%.:/();$=><|{}^\'`_]"
    # Utilisation d'une expression régulière pour remplacer tous les caractères de ponctuation
    return text.str.replace(punctuations, "", regex=True)


##########################################################################################################
# Suppression des emojis
def delete_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


##########################################################################################################
# Suppression des emojis


def raccourcir_texte(texte, longueur_max=500):
    return shorten(texte, width=longueur_max, placeholder="...")


#########################################################################################################
def detect_language(text):
    try:
        if text.strip():
            return detect(text)
        else:
            return "Erreur"
    except LangDetectException:
        return "Erreur"


"""
# Correction texte en français => Faux, ne pas prendre en compte
def correct_fr(text):
    nlp_fr = spacy.load("fr_core_news_sm")
    doc = nlp_fr(text)
    corrected_text = [token.text for token in doc]
    return " ".join(corrected_text)



# Correction texte en Allemand =>Faux, ne pas prendre en compte
def correct_de(text):
    nlp_de = spacy.load("de_core_news_sm")
    doc = nlp_de(text)
    corrected_text = [token.text for token in doc]
    return " ".join(corrected_text)


# Définissez une fonction pour la lemmatisation
def lemmatize_text(text):
    words = text.split()  # Divisez le texte en mots
    lemmatized_words = [
        lemmatizer.lemmatize(word) for word in words
    ]  # Lemmatisez chaque mot
    lemmatized_text = " ".join(lemmatized_words)  # Rejoignez les mots en un texte
    return lemmatized_text
"""
