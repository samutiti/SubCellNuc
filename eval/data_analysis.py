import anndata as ad 
import scanpy as sc

training_versions = [5]
FILTER = 'U2OS'
NUM_PCS = 50
LEI_RES = 0.3

for training_v in training_versions:
    if filter is None:
        embeddings = ad.read_h5ad(f"/scratch/users/samutiti/U54/SubCellNuc/training_V0{training_v}/inference.h5ad")
    else: embeddings = ad.read_h5ad(f"/scratch/users/samutiti/U54/SubCellNuc/training_V0{training_v}/inference_{FILTER}.h5ad")

    print(embeddings)
    adata = embeddings

    sc.pp.pca(
            adata,
            n_comps=NUM_PCS,
            svd_solver="arpack"
        )
    print('pca complete')

    sc.pp.neighbors(
            adata,
            n_neighbors=30,
            n_pcs=NUM_PCS,
        )
    print('nieghbors complete')

    sc.tl.leiden(
            adata,
            resolution=LEI_RES,
            key_added=f'leiden_{LEI_RES}'
        )
    print('leiden complete')

    sc.tl.umap(
            adata,
            min_dist=0.1,
        )

    print('umap complete')

    adata.obs["umap_x"] = adata.obsm["X_umap"][:, 0]
    adata.obs["umap_y"] = adata.obsm["X_umap"][:, 1]

    print(adata)
    if filter is None:
        adata.write_h5ad(f'/scratch/users/samutiti/U54/SubCellNuc/training_V0{training_v}/inference_analyzed.h5ad')
    else: adata.write_h5ad(f'/scratch/users/samutiti/U54/SubCellNuc/training_V0{training_v}/inference_{FILTER}_analyzed.h5ad')
    print('adata written')


    ### Visualize Umap
    save_suf = f"_training_V0{training_v}_mlp_embed.png" if not FILTER else f"_training_V0{training_v}__{FILTER}_mlp_embed.png"
    sc.pl.umap(
            adata,
            color=[f"leiden_{LEI_RES}"],
            show=False,
            save=save_suf
        )