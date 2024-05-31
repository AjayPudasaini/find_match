import os
import pandas as pd
import tensorflow as tf
import re, phonetics, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from utils.get_data import datas


import warnings
warnings.filterwarnings("ignore")


class DuplicateFinder:
    def __init__(self, dataframe: pd.DataFrame, model_path: str):
        self.df = dataframe
        self.model_path = model_path
        self.boolean_df = self.df.eq("")
        self._preprocess_data()
        self.vectorizerd_info = self._fit_vectorizers()
        self.weight_mapping_neural_network = tf.keras.models.load_model(self.model_path)

    def _preprocess_data(self):
        self.df["Citizenship_no"] = self.df["Citizenship_no"].astype(str)
        self.df['Date_of_birth'] = self.df['Date_of_birth'].str.replace(r'[-/]', '', regex=True)
        self.df['Citizenship_no'] = self.df['Citizenship_no'].str.replace(r'[-/]', '', regex=True)
        self.df['NAME_Preprocessed'] = self.df['Name'].str.split().apply(lambda x: " ".join(map(phonetics.metaphone, x)))
        self.df['FatherName_Preprocessed'] = self.df['Father_Name'].str.split().apply(lambda x: " ".join(map(phonetics.metaphone, x)))
        
        self.citizenship_numbers = self.df['Citizenship_no'].values
        self.date_of_birth_values = self.df['Date_of_birth'].values
        self.boolean_name = self.boolean_df["Name"].values
        self.boolean_father_name = self.boolean_df["Father_Name"].values

    def _fit_vectorizers(self):
        person_name_vectorizer = TfidfVectorizer()
        person_name_vectorizer.fit(self.df['NAME_Preprocessed'])
        person_name_vectorized = person_name_vectorizer.transform(self.df['NAME_Preprocessed'])

        father_name_vectorizer = TfidfVectorizer()
        father_name_vectorizer.fit(self.df['FatherName_Preprocessed'])
        father_name_vectorized = father_name_vectorizer.transform(self.df['FatherName_Preprocessed'])

        return {
            "NAME_Preprocessed": [person_name_vectorized, person_name_vectorizer],
            "FatherName_Preprocessed": [father_name_vectorized, father_name_vectorizer]
        }

    def _get_similarity_for_names(self, name: str, key: str) -> np.array:
        try:
            vectorized_info, vectorizer = self.vectorizerd_info[key]
            new_sim = cosine_similarity(vectorizer.transform([name]), vectorized_info).reshape(-1)
            new_sim = 0.5 * (new_sim + 1)  # Rescale from [-1, 1] to [0.5, 1]
            new_sim = (new_sim - 0.5) / 0.5  # Rescale from [0.5, 1] to [0, 1]
            if key == "NAME_Preprocessed":
                new_sim = np.where(self.boolean_name, np.nan, new_sim)
            elif key == "FatherName_Preprocessed":
                new_sim = np.where(self.boolean_father_name, np.nan, new_sim)
            return new_sim
        except Exception as e:
            print(f"An exception occurred in function get_similariy_for_names: {e}")

    def _get_similarity_for_DOB_CitizenshipNO(self, value: str, key: str) -> np.array:
        try:
            array_to_compare = self.citizenship_numbers if key == "Citizenship_no" else self.date_of_birth_values
            mask = array_to_compare != ""
            distances = np.where(mask, np.vectorize(Levenshtein.distance)(value, array_to_compare), np.nan)
            similarities = 1 - distances / np.maximum(len(value), np.vectorize(len)(array_to_compare))
            return similarities
        except Exception as e:
            print(f"An exception occurred in function get_similarity_for_DOB_CitizenshipNO: {e}")

    def _return_weighted_similarity(self, similarity_matrix: pd.DataFrame) -> pd.DataFrame:
        try:
            weight_matrix = ~similarity_matrix.isnull()
            weight_matrix = weight_matrix.astype(int)
            weight_matrix = np.round(self.weight_mapping_neural_network.predict(weight_matrix, batch_size=weight_matrix.shape[0], verbose=0), decimals=1)
            weight_matrix = np.multiply(similarity_matrix.values, weight_matrix)
            weighted_similarity = np.nansum(weight_matrix, axis=1)
            weight_matrix = pd.DataFrame(weight_matrix, columns=similarity_matrix.columns)
            weight_matrix['weighted_similarity'] = weighted_similarity
            return weight_matrix
        except Exception as e:
            print(f"An exception occurred in function return_weighted_similarity: {e}")

    def find_duplicates(self, threshold: float = 90.0) -> List[Dict[str, Any]]:
        subset_df = self.df[["id", "NAME_Preprocessed", "Citizenship_no", "Date_of_birth", "FatherName_Preprocessed"]].copy()
        final_result = []
        similarity_scores = []

        for current_row_index in range(len(subset_df)):
            similarity_values = {}
            row = subset_df.iloc[current_row_index].to_dict()
            for key, val in row.items():
                if key == "id":
                    continue
                if val == np.nan or val == "":
                    similarity_values[key] = np.repeat(np.nan, self.df.shape[0])
                else:
                    if key in ['NAME_Preprocessed', 'FatherName_Preprocessed']:
                        similarity_values[key] = self._get_similarity_for_names(val, key)
                    else:
                        similarity_values[key] = self._get_similarity_for_DOB_CitizenshipNO(val, key)
                        
            similarity_values = pd.DataFrame(similarity_values)[["NAME_Preprocessed", "Citizenship_no", "FatherName_Preprocessed", "Date_of_birth"]]
            similarity_values = self._return_weighted_similarity(similarity_values)
            similarity_values['weighted_similarity'] = similarity_values['weighted_similarity'] * 100  # Scale to percentage
            mask = similarity_values['weighted_similarity'] > threshold  # Use a threshold to identify similar entries
            duplicate_indexes = self.df.reset_index()[mask].index.to_list()  # Reset index before applying the mask

            duplicates = []
            for i, val in enumerate(duplicate_indexes):
                duplicates.append((current_row_index, val))
                similarity_scores.append(similarity_values['weighted_similarity'].iloc[val])

            while (current_row_index, current_row_index) in duplicates:
                duplicates.remove((current_row_index, current_row_index))
                
            unique_tuples = set()

            for t in duplicates:
                normalized_tuple = tuple(sorted(t))
                unique_tuples.add(normalized_tuple)

            result = list(unique_tuples)
            final_result += result

        final_result = list(set(final_result))
        
        final_output = []

        for pair in final_result:
            primary_customer_index, similar_screening_index = pair
            primary_customer_id = self.df.at[primary_customer_index, "id"]
            similar_screening_id = self.df.at[similar_screening_index, "id"]
            weighted_similarity = max(similarity_scores[final_result.index(pair)] for pair in final_result if primary_customer_index in pair or similar_screening_index in pair)
            
            final_output.append({
                "primary_customer_id": primary_customer_id,
                "similar_screening_id": similar_screening_id,
                "weighted_similarity": f"{weighted_similarity:.0f}"
            })

        return final_output


# Usage Example
# df = pd.DataFrame({
#     "Name": ["Sujan Neupane", "Sulav Neupane", "", "Ranjit Adhikari", "Sushil Karky", "Susheel karkey", "Sujan Neupane", "Sujan Neupane"],
#     "Citizenship_no" : ['1234567899', np.nan, '567866878712', '1213342311', "523121344", "523121344", "1234567899", "1234567899"],
#     "Date_of_birth": ['2020-01-01', '2010-05-29', "", '1999-09-09', '2001-01-01', "2001-01-01", "2020-01-01", "2020-01-01"],
#     "Father_Name": ["Sushil Neupane", "Sushil Neupane", "Utsav Pandey", "Ankit pandey", "Sushil Karkey", "Sushil Karkey", "Sushil Neupane", "Sushil Neupane"]
# })

df = datas()

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'Weight_Mapping_NN.keras')

finder = DuplicateFinder(df, model_path)
final_output = finder.find_duplicates()
