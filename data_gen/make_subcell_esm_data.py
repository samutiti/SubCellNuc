# load IFv25 file with all embeddings and gene names
# use gene names to get uniprot accession ids (subcell_embeddding, uniprot_id)
# load esm embeddings for all uniprot accession ids (in file)
# pair esm and subcell embeddings (subcell_embedding, esm_embedding)
# save as .npy file
import numpy as np
import pandas as pd
import torch
from unipressed import IdMappingClient


client = IdMappingClient()

def query_uniprot(gene_name): 
    response = client.search(gene_name, from_db='gene', to_db='uniprot')
    return [entry['to'] for entry in response] if response else [None]


embeddings, hpa_df = torch.load('data/all_harmonized.pt') # loads embeddings and dataframe and hpa
esm_embeddings = pd.read_csv('data/esm_embeddings.csv') # loads esm embeddings for all uniprot ids in file
# loop through embeddings, access protein name using hpa_df['gene_name'], query unitprot with gene names
for i, (emb, gene_name) in enumerate(zip(embeddings, hpa_df['gene_name'])):
    # query uniprot with gene name to get uniprot id
    # load esm embedding for uniprot id
    # pair subcell embedding and esm embedding
    # save as .npy file
    gene_names = gene_name.split(',') # some entries have multiple gene names separated by ','
    esm_embeddings = []
    uniprot_ids = query_uniprot(gene_names) # get uniprot ids for all gene names
    for uid in uniprot_ids:
        esm_embeddings.append(esm_embeddings[esm_embeddings['Unnamed: 0'] == uid].iloc[0, 1:] )
    esm_embeddings = np.array(esm_embeddings) # convert list of esm embeddings to numpy array

all_data = (embeddings, esm_embeddings) # implement this function to pair subcell and esm embeddings and save as .npy file
torch.save(all_data, 'subcell_esm_embeddings.pt') # save as .pt file