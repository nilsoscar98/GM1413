import os
import numpy as np
import pandas as pd
import matplotlib as plt
import pickle
import spacy
import concurrent.futures


#Function that reads all the pickle files in a folder, returns list of file names.
def read_file_names(folder_path: str) -> list[str]:
    files = [f for f in os.listdir(folder_path) if f.endswith('.p')]
    
    return files

def extract_data(folder_path: str, ticker: str) -> list[dict]:
    file_path = os.path.join(folder_path, ticker)  # File path.

    with open(file_path, 'rb') as file:  # Open file.
        content = pickle.load(file)  # Load content.
        data = []  # Create empty list.

        for entry in content:  # Iterate over the content.
            data.append(entry)  # Collect the dictionary.

    return data  # Return the list of dictionaries. 

def extract_and_process_transcript(transcripts: dict) -> list[str]:
    text_transcript = ''
    nlp = spacy.load('en_core_web_sm') #Load the spaCy model.
    for speech in transcripts:
        text_transcript += ' ' + speech['speech'][0]

    processed = [word.lemma_.lower() for word in nlp(text_transcript) if not (word.is_space or word.is_stop or word.is_punct)]
    
    return processed

def process_entry(entry):
    transcript = entry['transcript']
    processed = extract_and_process_transcript(transcript)
    return processed, len(processed)

def data_prep(in_data: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    num_rows = len(in_data)
    num_cols = max(len(row) for row in in_data)
    
    processed_speeches = np.empty((num_rows, num_cols), dtype=object)
    length_speeches = np.empty((num_rows, num_cols), dtype=int)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(num_rows):
            results = list(executor.map(process_entry, in_data[i]))
            for j, (processed, length) in enumerate(results):
                processed_speeches[i, j] = processed
                length_speeches[i, j] = length
    
    return processed_speeches, length_speeches



if __name__ == "__main__":
    folder_path = 'tickers' #Path to the folder containing the pickle files.
    files = read_file_names(folder_path) #List of all the pickle files in the folder.
    nlp = spacy.load('en_core_web_sm') #Load the spaCy model.
    in_data = [] #Initialize an empty list to store the transcripts.

    #Loop through all files and extract transcripts.
    for file in files[0:1]:
        transcripts = extract_data(folder_path, file)
        in_data.append(transcripts)
    #for file in files:
    #    transcripts = extract_data(folder_path, file) #Extract the transcripts.
    #    in_data.append(transcripts) #Append the transcripts to the list.

    processed_speeches, length_speeches = data_prep(in_data) #Process the speeches.

    df_processed_speeches = pd.DataFrame(processed_speeches)
    df_length_speeches = pd.DataFrame(length_speeches)
    df_combined = pd.concat([df_processed_speeches, df_length_speeches], axis=1)

    #Save the processed speeches to a CSV file.
    df_combined.to_excel('processed_speeches.xlsx', index=False) 






