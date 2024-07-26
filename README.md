# Mc-Donalds-Casestudy
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    mcdonalds = pd.read_csv('path_to_mcdonalds_dataset.csv')  
    print(mcdonalds.columns.tolist())
     print(mcdonalds.shape)
        print(mcdonalds.head(3))  
    MD_x = mcdonalds.iloc[:, :11].values
    MD_x = (MD_x == "Yes").astype(int)
    result = np.round(np.mean(MD_x, axis=0), 2)
    result
    MD_x = pd.DataFrame(...)  

    pca = PCA()
    MD_pca = pca.fit_transform(MD_x)

    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    np.random.seed(1234)


    best_model = None
    best_score = -1

    for n_clusters in range(2, 9):
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
    model.fit(MD.x)
    score = silhouette_score(MD.x, model.labels_)
    
    if score > best_score:
        best_score = score
        best_model = model

    MD_km28_labels = best_model.labels_



    summary_df = pd.DataFrame({
    'Standard deviation': np.sqrt(explained_variance),
    'Proportion of Variance': explained_variance_ratio,
    'Cumulative Proportion': cumulative_variance
    })

    print(summary_df)
    plt.plot(MD.m28)
    plt.ylabel("value of information criteria (AIC, BIC, ICL)")
    plt.show()
    MD_m4 = MD.m28.getModel(which="4")
    kmeans_clusters = MD.k4.clusters()
    mixture_clusters = MD_m4.clusters()

     result = pd.crosstab(kmeans_clusters, mixture_clusters)
     print(result)

     MD_m4a = GaussianMixture(n_components=4, random_state=0)
    MD_m4a.fit(MD.x, y=MD.k4.clusters())
    mixture_clusters = MD_m4a.predict(MD.x)

    kmeans_clusters = MD.k4.clusters()

    result = pd.crosstab(kmeans_clusters, mixture_clusters)
     print(result)
     log_lik_m4a = MD_m4a.score(MD.x) * MD.x.shape[0]
     log_lik_m4 = MD_m4.score(MD.x) * MD.x.shape[0]

     print(f"'log Lik.' {log_lik_m4a} (df=47)")
    print(f"'log Lik.' {log_lik_m4} (df=47)")
    rev(table(mcdonalds$Like))
    mcdonalds$Like.n <- 6 - as.numeric(mcdonalds$Like)
    table(mcdonalds$Like.n)
    f <- paste(names(mcdonalds)[1:11], collapse = "+")
     f <- paste("Like.n ~ ", f, collapse = "")
     f <- as.formula(f)
     f
     set.seed(1234)
     MD.reg2 <- stepFlexmix(f, data = mcdonalds, k = 2, nrep = 10, verbose = FALSE)
    MD.reg2
