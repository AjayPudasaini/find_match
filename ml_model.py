# ml_model.py
import os, phonetics, warnings, Levenshtein, tensorflow as tf, numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings('ignore')

# Sample DataFrame
df = pd.DataFrame({
    "Name": ["Sujan Neupane", "Sulav Neupane", "", "Ranjit Adhikari", "Sushil Karky", "Susheel karkey"],
    "Citizenship_no": ['1234567899', np.nan, '567866878712', '1213342311', "523121344", "523121344"],
    "Date_of_birth": ['2020-01-01', '2010-05-29', '', '1999-09-09', '2001-01-01', "2001-01-01"],
    "Father_Name": ["Sushil Neupane", "Ranjit Neupane", "Utsav Pandey", "Ankit pandey", "Sushil Karkey", "Sushil Karkey"]
})

# Preprocess data
boolean_df = df.eq("")
df["Citizenship_no"] = df["Citizenship_no"].astype(str)
df['Date_of_birth'] = df['Date_of_birth'].str.replace(r'[-/]', '', regex=True)
df['Citizenship_no'] = df['Citizenship_no'].str.replace(r'[-/]', '', regex=True)
df['NAME_Preprocessed'] = df['Name'].str.split().apply(lambda x: " ".join(map(phonetics.metaphone, x)))
df['FatherName_Preprocessed'] = df['Father_Name'].str.split().apply(lambda x: " ".join(map(phonetics.metaphone, x)))

person_name_vectorizer = TfidfVectorizer()
person_name_vectorizer.fit(df['NAME_Preprocessed'])
person_name_vectorized = person_name_vectorizer.transform(df['NAME_Preprocessed'])

father_name_vectorizer = TfidfVectorizer()
father_name_vectorizer.fit(df['FatherName_Preprocessed'])
father_name_vectorized = father_name_vectorizer.transform(df['FatherName_Preprocessed'])

vectorizerd_info = {
    "NAME_Preprocessed": [person_name_vectorized, person_name_vectorizer],
    "FatherName_Preprocessed": [father_name_vectorized, father_name_vectorizer]
}

citizenship_numbers = df['Citizenship_no'].values
date_of_birth_values = df['Date_of_birth'].values
boolean_name = boolean_df["Name"].values
boolean_father_name = boolean_df["Father_Name"].values

# Load the neural network model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'Weight_Mapping_NN.keras')
print(f"Loading model from: {model_path}")
assert os.path.exists(model_path), "Model file does not exist!"

try:
    weight_mapping_neural_network = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    raise ValueError(f"Failed to load model: {e}")

def get_similariy_for_names(name: str, key: str) -> np.array:
    try:
        vectorized_info, vectorizer = vectorizerd_info[key]
        new_sim = cosine_similarity(vectorizer.transform([name]), vectorized_info).reshape(-1)
        new_sim = 0.5 * (new_sim + 1)  # Rescale from [-1, 1] to [0.5, 1]
        new_sim = (new_sim - 0.5) / 0.5  # Rescale from [0.5, 1] to [0, 1]
        if key == "NAME_Preprocessed":
            new_sim = np.where(boolean_name, np.nan, new_sim)
        elif key == "FatherName_Preprocessed":
            new_sim = np.where(boolean_father_name, np.nan, new_sim)
        return new_sim
    except Exception as e:
        print(f"An exception occurred in function get_similariy_for_names: {e}")

def get_similarity_for_DOB_CitizenshipNO(value: str, key: str) -> np.array:
    try:
        array_to_compare = citizenship_numbers if key == "Citizenship_no" else date_of_birth_values
        mask = array_to_compare != ""
        distances = np.where(mask, np.vectorize(Levenshtein.distance)(value, array_to_compare), np.nan)
        similarities = 1 - distances / np.maximum(len(value), np.vectorize(len)(array_to_compare))
        return similarities
    except Exception as e:
        print(f"An exception occurred in function get_similarity_for_DOB_CitizenshipNO: {e}")


def return_weighted_similarity(similarity_matrix: pd.DataFrame) -> pd.DataFrame:
    try:
        weight_matrix = ~similarity_matrix.isnull()
        weight_matrix = weight_matrix.astype(int)
        weight_matrix = np.round(weight_mapping_neural_network.predict(weight_matrix, batch_size=weight_matrix.shape[0], verbose=0), decimals=1)
        weight_matrix = np.multiply(similarity_matrix.values, weight_matrix)
        weighted_similarity = np.nansum(weight_matrix, axis=1)
        weight_matrix = pd.DataFrame(weight_matrix, columns=similarity_matrix.columns)
        weight_matrix['weighted_similarity'] = weighted_similarity
        return weight_matrix
    except Exception as e:
        print(f"An exception occurred in function return_weighted_similarity: {e}")


def process_similarity():
    subset_df = df[["NAME_Preprocessed", "Citizenship_no", "Date_of_birth", "FatherName_Preprocessed"]].copy()
    final_result = []

    for current_row_index in range(len(subset_df)):
        similarity_values = {}
        row = subset_df.iloc[current_row_index].to_dict()
        for key, val in row.items():
            if val == np.nan or val == "":
                similarity_values[key] = np.repeat(np.nan, df.shape[0])
            else:
                if key in ['NAME_Preprocessed', 'FatherName_Preprocessed']:
                    similarity_values[key] = get_similariy_for_names(val, key)
                else:
                    similarity_values[key] = get_similarity_for_DOB_CitizenshipNO(val, key)
                    
        similarity_values = pd.DataFrame(similarity_values)[["NAME_Preprocessed", "Citizenship_no", "FatherName_Preprocessed", "Date_of_birth"]]
        similarity_values = return_weighted_similarity(similarity_values)
        mask = similarity_values['weighted_similarity'] > 0.9
        duplicae_indexes = df[mask].index.to_list()
        duplicates = [(current_row_index, val) for val in duplicae_indexes]

        while (current_row_index, current_row_index) in duplicates:
            duplicates.remove((current_row_index, current_row_index))
            
        unique_tuples = {tuple(sorted(t)) for t in duplicates}
        result = list(unique_tuples)
        final_result += result
    
    final_result = list(set(final_result))
    return final_result
