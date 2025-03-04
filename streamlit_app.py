import streamlit as st
from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity # for fuzzy match
import numpy as np

# Show title and description.
st.title("📄 RBM Case Study Retrieval")
st.write(
    "Input a topic and GPT will provide the best RBM case study! "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Fuzzy Matching Functions
def get_embedding(text, model="text-embedding-3-small"):
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)
    text = str(text)
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding
    
def compute_cosine_similarity(base_string, string_list, list_embeddings):
    model = "text-embedding-3-small"
    base_embedding = get_embedding(base_string, model)
    
    # Convert embeddings to numpy arrays
    base_embedding = np.array(base_embedding).reshape(1, -1)
    
    # Compute cosine similarity between the base string and all other strings
    similarity_scores = cosine_similarity(base_embedding, list_embeddings).flatten()

    return similarity_scores

def fuzzy_match(base_string, string_list, list_embeddings):
    '''
    How it works:
    Loading the Model: The script fetches the get_openai_embedding function fetches the embedding for a given text.
    Encoding Strings: The base string and each string in the list are encoded into high-dimensional vectors using the model.
    Cosine Similarity Computation: The cosine similarity between the base string vector and each vector in the string list is computed.
    Results Display: The similarity scores are printed for each string in the list relative to the base string. The script also identifies the string with the highest similarity score.
    '''
    output = ""
    # check whether the item exists exactly
    if base_string in string_list:
        output = base_string
    else:
        similarity_scores = compute_cosine_similarity(base_string, string_list, list_embeddings)
    
        # To find the most similar string
        most_similar_index = np.argmax(similarity_scores)
        output = string_list[most_similar_index]
        
    return output, most_similar_index

def generate_embeddings_case_studies(df):
    # df["combined_text"] = df["case study"].astype(str) + " " + df["applications"] + " " + df["content"].astype(str)
    df["embeddings"] = df["combined_text"].apply(get_embedding)
    return df

def retrieve_case_study(query, df):
    output, index = fuzzy_match(query, df['case study'].tolist(), df["embeddings"].tolist())
    return f"Case Study - Description: {output}. Link here: {df.iloc[index]['read more']}" , 



# Load the excel list and re-write the embeddings
df_case_studies = pd.read_excel('RBM_case_studies.xlsx', index_col=0)
df_case_studies = generate_embeddings_case_studies(df_case_studies)


# Ask the user for a question via `st.text_area`.
question = st.text_area(
    "Now give me details about a topic or commercial use case.",
    placeholder="Can you give me a short summary?"
)

if question:

    output = retrieve_case_study(question, df_case_studies)

    # output the response to the app using `st.write_stream`.
    st.write(output)
