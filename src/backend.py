"""Helper module. Imports the models, dataframes and define processing functions. Caches most files to improve performance."""
# A cautious computing machine walks into a bar... 

import src.config as config
import pandas as pd
from cv2 import imread, resize, cvtColor, IMREAD_COLOR, COLOR_BGR2RGB
from numpy import argmax
import streamlit as st
from numpy import expand_dims
from keras.saving import load_model
from numpy import zeros, float32
import os
import joblib
import nltk
from keras.preprocessing.sequence import pad_sequences

# Data ingestion functions - Cached for optimization
@st.cache_data
def load_processed_df():
	"""Load the processed dataframe."""
	return pd.read_csv(config.dataframe_path)

@st.cache_data
def load_raw_df():
	"""Returns the unprocessed dataframe. Returns a cached (therefore immutable) object"""
	return pd.read_csv(config.raw_dataframe_path, sep=',', index_col=0)

@st.cache_data
def load_y_df():
	"""Returns y"""
	return pd.read_csv(config.y_path, sep=',', index_col='Unnamed: 0')

@st.cache_data
def load_df_no_processing():
	"""Returns the unprocessed dataframe joined to y"""
	return load_raw_df().join(load_y_df())


@st.cache_data
def load_label_encoder():
	"""Returns the label encoder"""
	return joblib.load(config.label_encoder_path)

@st.cache_resource
def load_multimodal_classifier():
	"""Returns the multimodal classifier"""
	return load_model(config.multimodal_model_path)

@st.cache_resource
def load_lstm():
	return load_model("data/lstm_model.keras")

@st.cache_resource
def load_linearsvc_classifier():
	"""For caching purpose. Do not call."""
	return joblib.load(config.linearsvc_model_path)


@st.cache_data
def load_tfidf_vectorizer():
	"""For caching purpose. Do not call."""
	return joblib.load(config.tfidf_path)

@st.cache_data
def load_tokenizer():
	"""For caching purpose. Do not call."""
	return joblib.load(config.tokenizer_path)

@st.cache_data
def load_vectorizer():
	"""Returns the LSTM trained vectorizer"""
	return joblib.load(config.vectorizer_path)


@st.cache_data
def load_tokenizer():
	""""""
	return joblib.load(config.tokenizer_path)


# Data treatment functions

@st.cache_data
def predict_with_multimodal(processed_image, processed_text):
	"""Runs the multimodal classifier on the inputs. Takes advantage of data caching."""
	image = expand_dims(cvtColor(resize(processed_image, (200,200)), COLOR_BGR2RGB), axis=0)
	preds = load_multimodal_classifier().__call__(inputs=(image, processed_text))
	return load_label_encoder().inverse_transform([argmax(preds)])

@st.cache_data
def predict_with_lstm(text):
	return load_lstm().__call__(pad_text(vectorize_text(text)))


# These two functions could realistically benefit from caching as their origin page can be 
# reloaded multiple times without changing the data
@st.cache_data
def vectorize_text(text):
	"""Vectorize a text sequence. Cached."""
	if isinstance(text, int):
		return []
	return load_vectorizer().texts_to_sequences(text)

@st.cache_data 
def pad_text(text):
	"""Pad a tokenized text sequence. Cached"""
	return pad_sequences(text, maxlen=527)


# Caching gives out some scope error, too late to fix so I disabled it
def fetch_unprocessed_images(namelist):
	"""Read images in the input from the raw images folder. Returned images are in  500x500 BGR float32 format, with values in [0,1]"""
	output = zeros((len(namelist), 500, 500, 3))
	error_happened = False
	for i in range(len(namelist)):
		try:
			if os.path.exists(config.raw_img_folder+"/"+namelist[i]):
				output[i] = imread(config.raw_img_folder+"/"+namelist[i], IMREAD_COLOR)
			else:
				error_happened = True
		except:
			error_happened = True
	if output.max() > 1 :
		output /= 255
	if error_happened:
		st.toast("Un problème est survenu lors du chargement des images...", icon="❓")
	return float32(output)

# These ones most likely won't ever be called on the same arguments, so caching will 
# only grow the memory stack without giving any performance improvements
def fetch_processed_images(namelist):
	"""Read images in the input from the preprocessed images folder. Images class folder need to be appended before the image name. Returned images are in 500x500 BGR float32 format, with values in [0,1]"""
	output = zeros((len(namelist), 500, 500, 3))
	error_happened = False
	for i in range(len(namelist)):
		try:
			if os.path.exists(config.preprocessed_img_folder.format("train")+"/"+namelist[i]):
				output[i] = imread(config.preprocessed_img_folder.format("train")+"/"+namelist[i], IMREAD_COLOR)
			elif os.path.exists(config.preprocessed_img_folder.format("test")+"/"+namelist[i]):
				output[i] = imread(config.preprocessed_img_folder.format("test")+"/"+namelist[i], IMREAD_COLOR)
			elif os.path.exists(config.preprocessed_img_folder.format("discarded")+"/"+namelist[i]):
				output[i] = imread(config.preprocessed_img_folder.format("discarded")+"/"+namelist[i], IMREAD_COLOR)
			else:
				error_happened = True
		except:
			error_happened = True
	if output.max() > 1 :
		output /= 255
	if error_happened:
		st.toast("Un problème est survenu lors du chargement des images", icon="❓")
	return float32(output)
			
def df_to_processed_images(df):
	"""Returns the image names of the input dataframe while appending the class folder before"""
	return df["prdtypecode"] +"/image_" + df["imageid"].astype(str) + "_product_" + df["productid"].astype(str) + ".jpg"

def df_to_raw_images(df):
	"""Returns the image names of the input dataframe without appending the class folder before"""
	return "image_" + df["imageid"].astype(str) + "_product_" + df["productid"].astype(str) + ".jpg"

# Variable declarations, poor man singleton
df=load_processed_df()
#nltk.download('punkt')