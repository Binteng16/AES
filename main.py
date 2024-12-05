import streamlit as st
import pandas as pd

# Streamlit title and description
st.title("Essay Dataset Processing")
st.write("""
This application allows you to upload a `.tsv` file, process the data, 
and normalize specific scores for structure and grammar. The processed dataset 
can then be downloaded in CSV format.
""")

# File upload
uploaded_file = st.file_uploader("Upload your .tsv file", type="tsv")

# Proceed if the file is uploaded
if uploaded_file is not None:
    # Read the dataset from the uploaded file
    raw_data = pd.read_csv(uploaded_file, sep='\t', encoding='ISO-8859-1')
    
    # Show the first few rows of the dataset for inspection
    st.subheader("Raw Data Preview")
    st.write(raw_data.head())
    
    # Filter the dataset for essay_set = 7
    filtered_dataset = raw_data[raw_data['essay_set'] == 7].copy()

    # Define the columns for Structure and Grammar scores
    skor_struktur = ['rater1_trait1', 'rater1_trait2', 'rater1_trait3', 'rater2_trait1', 'rater2_trait2', 'rater2_trait3']
    skor_tata_bahasa = ['rater1_trait4', 'rater2_trait4']

    # Calculate Structure score as the average of 6 traits
    filtered_dataset['skor_struktur'] = filtered_dataset[skor_struktur].sum(axis=1) / len(skor_struktur)

    # Calculate Grammar score as the average of 2 traits
    filtered_dataset['skor_tata_bahasa'] = filtered_dataset[skor_tata_bahasa].sum(axis=1) / len(skor_tata_bahasa)

    # Normalize Structure score between 0 and 10
    min_value_struktur = filtered_dataset['skor_struktur'].min()
    max_value_struktur = filtered_dataset['skor_struktur'].max()
    filtered_dataset['skor_struktur_normalized'] = 10 * (filtered_dataset['skor_struktur'] - min_value_struktur) / (max_value_struktur - min_value_struktur)
    filtered_dataset['skor_struktur_normalized'] = filtered_dataset['skor_struktur_normalized'].round(1)

    # Normalize Grammar score between 0 and 10
    min_value_tata_bahasa = filtered_dataset['skor_tata_bahasa'].min()
    max_value_tata_bahasa = filtered_dataset['skor_tata_bahasa'].max()
    filtered_dataset['skor_tata_bahasa_normalized'] = 10 * (filtered_dataset['skor_tata_bahasa'] - min_value_tata_bahasa) / (max_value_tata_bahasa - min_value_tata_bahasa)
    filtered_dataset['skor_tata_bahasa_normalized'] = filtered_dataset['skor_tata_bahasa_normalized'].round(1)

    # Select relevant columns for the processed data
    pre_processing_data = ['essay_id', 'essay_set', 'essay', 'skor_struktur_normalized', 'skor_tata_bahasa_normalized']
    final_pre_processing_data = filtered_dataset[pre_processing_data]

    # Show processed data
    st.subheader("Processed Data Preview")
    st.write(final_pre_processing_data.head())

    # Provide a button to download the processed data
    csv = final_pre_processing_data.to_csv(index=False)
    st.download_button(
        label="Download Processed Data",
        data=csv,
        file_name="pre_processing_data.csv",
        mime="text/csv"
    )

else:
    st.warning("Please upload a .tsv file to proceed.")
