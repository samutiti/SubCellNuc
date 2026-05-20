import anndata as ad 
import scanpy as sc

embedding_files = ['/scratch/users/samutiti/U54/SubCellNuc/training_V10/inference_U2OS_mean.h5ad']
prefixes = ['U2OS_mean']
versions = [10]
NUM_PCS = 50
LEI_RES = 0.3

for embedding_f, prefix, training_v in zip(embedding_files, prefixes, versions):
    adata = ad.read_h5ad(embedding_f)
    print(adata)

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
    adata.write_h5ad(f'/scratch/users/samutiti/U54/SubCellNuc/training_V{training_v}/{prefix}_inference_analyzed.h5ad')
    print('adata written')


    ### Visualize Umap
    save_suf = f"v{training_v}_{prefix}_mlp_embed.png"
    sc.pl.umap(
            adata,
            color=[f"leiden_{LEI_RES}"],
            show=False,
            save=save_suf
        )