# load IFv25 file with all embeddings and gene names
# use gene names to get uniprot accession ids (subcell_embeddding, uniprot_id)
# load esm embeddings for all uniprot accession ids (in file)
# pair esm and subcell embeddings (subcell_embedding, esm_embedding)
# save as .npy file
import numpy as np
import pandas as pd
import torch
from unipressed import IdMappingClient
from tqdm import tqdm


client = IdMappingClient()

def query_uniprot(gene_names): 
    response = client.submit(source="GeneCards", dest="UniProtKB", ids=set(gene_names))
    return [entry['to'] for entry in response.each_result()] if response else [None]


hpa_df, embeddings = torch.load('/scratch/users/samutiti/U54/embeddings/all_harmonized_features_microscope_vit.pth', weights_only=False) # loads embeddings and dataframe and hpa
# esm_embeddings = pd.read_csv('/scratch/users/samutiti/U54/embeddings/seq_emd.tsv', sep='\t') # loads esm embeddings for all uniprot ids in file
esm_embeddings = pd.read_csv('/scratch/users/samutiti/U54/embeddings/esm_mean_emb_df.csv')
esm_embeddings = esm_embeddings[esm_embeddings.columns[1:1282]]
# loop through embeddings, access protein name using hpa_df['gene_name'], query unitprot with gene names
for i, (emb, gene_name) in tqdm(enumerate(zip(embeddings, hpa_df['gene_names'])), total=len(embeddings)):
    # query uniprot with gene name to get uniprot id
    # load esm embedding for uniprot id
    # pair subcell embedding and esm embedding
    # save as .npy file
    gene_names = gene_name.split(',') # some entries have multiple gene names separated by ','
    curr_embeddings = []
    uniprot_ids = query_uniprot(gene_names) # get uniprot ids for all gene names
    for uid in uniprot_ids:
        curr_embeddings.append(esm_embeddings[esm_embeddings['gene'] == uid].iloc[0][1:])
    curr_embeddings = np.array(curr_embeddings) # convert list of esm embeddings to numpy array

all_data = (embeddings, curr_embeddings) # implement this function to pair subcell and esm embeddings and save as .npy file
torch.save(all_data, 'subcell_esm_embeddings.pt') # save as .pt file