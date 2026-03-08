# load IFv25 file with all embeddings and gene names
# use gene names to get uniprot accession ids (subcell_embeddding, uniprot_id)
# load esm embeddings for all uniprot accession ids (in file)
# pair esm and subcell embeddings (subcell_embedding, esm_embedding)
# save as .npy file
import numpy as np
import pandas as pd
import torch