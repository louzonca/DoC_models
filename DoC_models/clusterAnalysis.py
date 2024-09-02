from permetrics import ClusteringMetric

nneighb = 5
minSamp = 3
minSiz=3

clustcmap = plt.get_cmap('Spectral')
cm_dict={}
if model == ahp_model:
    param_list = [['X_eq', 'Y_eq','T_hyp','1/tmAHP', '1/tsAHP', 'sigma'], 
                  ['X_eq','T_hyp','1/tmAHP', '1/tsAHP', 'sigma'],
                  ['Y_eq','T_hyp','1/tmAHP', '1/tsAHP', 'sigma'],
                  ['X_eq', 'Y_eq','1/tmAHP','1/tsAHP', 'sigma'],
                  ['X_eq', 'Y_eq','T_hyp','1/tsAHP', 'sigma'],
                  ['X_eq', 'Y_eq','T_hyp','1/tmAHP', 'sigma'],
                  ['X_eq', 'Y_eq','T_hyp','1/tmAHP', '1/tsAHP']]#,
                  #['X_eq', 'Y_eq','T_hyp','1/tmAHP'],
                  #['X_eq', 'Y_eq','T_hyp','1/tsAHP'],
                  #['X_eq', 'Y_eq','T_hyp','sigma'],
                  #['X_eq', 'Y_eq','1/tmAHP','sigma'],
                  #['X_eq', 'Y_eq','1/tsAHP','sigma'],
                  #['X_eq','T_hyp','1/tmAHP', '1/tsAHP'],
                  #['X_eq','T_hyp','1/tmAHP', 'sigma'],
                  #['X_eq','T_hyp','1/tsAHP', 'sigma'],
                  #['X_eq','1/tmAHP', '1/tsAHP', 'sigma'],
                  #['X_eq', 'Y_eq'], 
                  #['X_eq', 'T_hyp'], 
                  #['X_eq', 'sigma'], 
                  #['X_eq', '1/tsAHP'], 
                  #['T_hyp','1/tmAHP', '1/tsAHP'],
                  #['X_eq','T_hyp', 'sigma'],
                  #['X_eq','1/tsAHP', 'sigma']]

elif model == hopf_model:
    param_list = [['a'], ['sigma'], ['a', 'sigma']]

metadata = pd.read_excel('fMRI_DoC_patients_forPD.xlsx')

for param_choice in param_list:
    figU, ax = plt.subplots(3,2, figsize=(35,35))
    iter = 2
    reducer = umap.UMAP(
        random_state=42,
        n_neighbors=nneighb,
        min_dist=0.1,
        n_components=2).fit(dfNorm[f'iter_{iteration}'][param_choice])
    embedding = reducer.transform(dfNorm[f'iter_{iteration}'][param_choice]) 
    embedding.shape

    labels = hdbscan.HDBSCAN(
        min_samples=minSamp,
        min_cluster_size=minSiz,
    ).fit_predict(embedding)

    clustered = (labels >= 0)

    nClusts = np.max(labels)+1
    clusts_cols = [clustcmap(x/(nClusts-1)) for x in range(nClusts)]  

    ax[0,0].scatter(
        embedding[~clustered, 0],
        embedding[~clustered, 1],
        c=(0.5,0.5,0.5),
        alpha=0.5)
    ax[0,0].scatter(
        embedding[clustered, 0],
        embedding[clustered, 1],
        c=[clusts_cols[x] for x in labels[clustered]])
    ax[0,0].set_title(str(param_choice))

    lab = hdbscan.HDBSCAN(
        min_samples=minSamp,
        min_cluster_size=minSiz,
    ).fit(embedding)

    lab.condensed_tree_.plot(select_clusters=True, selection_palette=clusts_cols, axis=ax[0,1])
    cm = ClusteringMetric(X=embedding[clustered,:], y_pred=labels[clustered])
    cm_dict[str(param_choice)]=cm.DBCVI()
    ax[0,1].set_title(f'DBVCI = {cm.DBCVI()}')

    age = {}
    genre = {}
    etiology = {}
    outcome = {}
    nClusts = np.max(labels)+1
    dfclusts = pd.DataFrame
    for l in range(nClusts):
        age[f'cluster_{l}'] = metadata.iloc[labels==l]['AGE']
        genre[f'cluster_{l}'] = metadata.iloc[labels==l]['GENRE']
        etiology[f'cluster_{l}'] = metadata.iloc[labels==l]['CODE_ETIOLOGIEclean']
        outcome[f'cluster_{l}'] = metadata.iloc[labels==l]['outcome']

    dfclustsMF = pd.DataFrame([sum(genre[f'cluster_{c}']=='M') for c in range(nClusts)], columns=['nM'])
    dfclustsMF.insert(1,'nF',[sum(genre[f'cluster_{c}']=='F') for c in range(nClusts)])
    dfclustsMF.insert(2, 'Cluster', [c for c in range(nClusts)])

    dfclustsEtio = pd.DataFrame([sum(etiology[f'cluster_{c}']=='anoxia') for c in range(nClusts)], columns=['nAnox'])
    dfclustsEtio.insert(1,'nTBI',[sum(etiology[f'cluster_{c}']=='TBI') for c in range(nClusts)])
    dfclustsEtio.insert(2,'nAnTBI',[sum(etiology[f'cluster_{c}']=='anoxia TBI') for c in range(nClusts)])
    dfclustsEtio.insert(3,'nSAH',[sum(etiology[f'cluster_{c}']=='SAH') for c in range(nClusts)])
    dfclustsEtio.insert(4,'nStroke',[sum(etiology[f'cluster_{c}']=='stroke') for c in range(nClusts)])
    dfclustsEtio.insert(5,'nOther',[sum(etiology[f'cluster_{c}']=='other') for c in range(nClusts)])
    dfclustsEtio.insert(6, 'Cluster', [c for c in range(nClusts)])

    dfclustsOutcome = pd.DataFrame([sum(outcome[f'cluster_{c}']=='dead') for c in range(nClusts)], columns=['nDead'])
    dfclustsOutcome.insert(1,'nConscious',[sum(outcome[f'cluster_{c}']=='conscient') for c in range(nClusts)])
    dfclustsOutcome.insert(2,'nStable',[sum(outcome[f'cluster_{c}']=='stable') for c in range(nClusts)])
    dfclustsOutcome.insert(3,'nUp',[sum(outcome[f'cluster_{c}']=='up') for c in range(nClusts)])
    dfclustsOutcome.insert(4,'nDown',[sum(outcome[f'cluster_{c}']=='down') for c in range(nClusts)])
    dfclustsOutcome.insert(5, 'Cluster', [c for c in range(nClusts)])


    # Age box plots
    bp = ax[1,0].boxplot([age[f'cluster_{i}'] for i in range(np.max(labels)+1)],
                    patch_artist=True, positions = [i for i in range(np.max(labels)+1)], widths = 0.5)
    style_box_plots(bp, clusts_cols)

    # Sex plot
    cols_et = plt.get_cmap('gnuplot')
    dfclustsMF.plot.bar(x='Cluster', stacked=True, ax=ax[1,1], color=[cols_et(1/10), cols_et(9/10)])

    # Etiology plot
    cdictEt = {'nAnox': cols_et(0/5), 'nAnTBI': cols_et(1/5), 'nTBI': cols_et(2/5), 'nSAH': cols_et(3/5), 'nOther': cols_et(4/5),  'nStroke': cols_et(5/5)}
    dfclustsEtio.plot.bar(x='Cluster', stacked=True, ax=ax[2,0], color=[cdictEt[x] for x in list(dfclustsEtio.columns)[0:-1]])

    # Outcome plot
    cols_out = plt.get_cmap('pink')
    cdictOut = {'nConscious': cols_out(4/6), 'nUp': cols_out(3/6), 'nStable': cols_out(2/6), 'nDown': cols_out(1/6), 'nDead': cols_out(0/6)}
    dfclustsOutcome.plot.bar(x='Cluster', stacked=True, ax=ax[2,1], color=[cdictOut[x] for x in list(dfclustsOutcome.columns)[0:-1]])
plt.show()


## ------- UMAPs successive iterations----------
cmapjet = plt.get_cmap('jet')
patnum = [str(i) for i in pat_list]
nneighb = 5
embedding = dict()
figC, ax = plt.subplots(1,1)
for iter in range(maxIter+1):
    if model.__name__ == 'hopf_model':
        reducer = umap.UMAP(
            random_state=42,
            n_neighbors=nneighb,
            min_dist=0.1,
            n_components=2).fit(df[f'iter_{iter}'][['a', 'sig']])
        embedding[f'iter_{iter}'] = reducer.transform(df[f'iter_{iter}'][['a', 'sig']])
    elif model.__name__ == 'ahp_model':
        reducer = umap.UMAP(
            random_state=42,
            n_neighbors=nneighb,
            min_dist=0.1,
            n_components=2).fit(df[f'iter_{iter}'][['X_eq', 'Y_eq','T_hyp','1/tmAHP', '1/tsAHP', 'sigma']])
        embedding[f'iter_{iter}'] = reducer.transform(df[f'iter_{iter}'][['X_eq', 'Y_eq','T_hyp','1/tmAHP', '1/tsAHP', 'sigma']])
    embedding[f'iter_{iter}'].shape

    #ax = plt.axes(projection="3d")
    ax.scatter(
        embedding[f'iter_{iter}'][:, 0],
        embedding[f'iter_{iter}'][:, 1],
        #embedding[:, 2],
        #s=df.Cluster*20,
        #c=[colors[cmap[x]] for x in df.Cluster])
        #c=[colors_gp_5[x] for x in df[f'iter_{iter}'].Group.map({"EMCS":0, "MCS+":1, "MCS-":2, "VS":3, "COMA":4})],
        #c=[cmapjet(i/50) for i in pat_list],
        alpha=1/(iter+1))
    for i, num in enumerate(patnum):
        ax.annotate(num, (embedding[f'iter_{iter}'][i,0], embedding[f'iter_{iter}'][i,1]))
for i, num in enumerate(patnum):
    for iter in range(maxIter):
        plt.plot([embedding[f'iter_{iter}'][i, 0], embedding[f'iter_{iter+1}'][i, 0]],
                 [embedding[f'iter_{iter}'][i, 1], embedding[f'iter_{iter+1}'][i, 1]],
                 #c=[colors_gp_5[x] for x in df[f'iter_{0}'].Group.map({"EMCS":0, "MCS+":1, "MCS-":2, "VS":3, "COMA":4})][i],
                 )#c=[cmapjet(i/50) for i in pat_list])
                #alpha=0.5)

figC.show()
