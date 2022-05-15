from sklearn.manifold import TSNE
import torch
import bitstring
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
from tqdm import tqdm
tqdm.pandas()

def main():

    reg = ['jiashenliu.515k-hotel-reviews-data-in-europe',
             'paolocons.another-fiat-500-dataset-1538-rows',
             'danielkyrka.bmw-pricing-challenge',
             'antfarol.car-sale-advertisements',
             'gagandeep16.car-sales',
             'austinreese.craigslist-carstrucks-data',
             'jpallard.google-store-ecommerce-data-fake-retail-data',
             'roccoli.gpx-hike-tracks',
             'rohitmathur100.happiness',
             'jmmvutu.summer-products-and-sales-in-ecommerce-wish',
             'nicapotato.womens-ecommerce-clothing-reviews',
             'unsdsn.world-happiness',
             'home-depot-product-search-relevance',
             'liberty-mutual-group-property-inspection-prediction',
             'machinery-tube-pricing',
             'rossmann-store-sales']

    cls = ['barelydedicated.bank-customer-churn-modeling',
             'abineshkumark.carsdata',
             'mathchi.churn-for-bank-customers',
             'sakshigoyal7.credit-card-customers',
             'christianlillelund.csgo-round-winner-classification',
             'uciml.default-of-credit-card-clients-dataset',
             'futurecorporation.epitope-prediction',
             'gdaley.hkracing',
             'asad1m9a9h6mood.news-articles',
             'septa.on-time-performance',
             'rounakbanik.pokemon',
             'terminus7.pokemon-challenge',
             'aaron7sun.stocknews',
             'rajeevw.ufcdata',
             'sobhanmoosavi.us-accidents',
             'bnp-paribas-cardif-claims-management',
             'crowdflower-search-relevance',
             'flavours-of-physics',
             'predict-west-nile-virus',
             'santander-customer-satisfaction']
    datasets_clusters={
            'unsdsn.world-happiness':'happiness',
            'rohitmathur100.happiness':'happiness',
            ###########
            'sakshigoyal7.credit-card-customers':'finance',
            'jpallard.google-store-ecommerce-data-fake-retail-data':'finance',
            'mathchi.churn-for-bank-customers':'finance',
            'barelydedicated.bank-customer-churn-model':'finance',
            'rossmann-store-sales':'finance',
        'uciml.default-of-credit-card-clients-dataset':'finance',
        'barelydedicated.bank-customer-churn-modeling':'finance',
        'santander-customer-satisfaction':'finance',
            ###########
            'asad1m9a9h6mood.news-articles':'news',
            'aaron7sun.stocknews':'news',
            ###########
            'machinery-tube-pricing':'sales',
            'danielkyrka.bmw-pricing-challenge':'sales',

            ###########
            'jiashenliu.515k-hotel-reviews-data-in-europe':'reviews',
            'nicapotato.womens-ecommerce-clothing-reviews':'reviews',
            'jmmvutu.summer-products-and-sales-in-ecommerce-wish':'reviews',    
            'crowdflower-search-relevance':'reviews',
            'home-depot-product-search-relevance':'reviews',
            ###########
            'abineshkumark.carsdata':'cars',
            'gagandeep16.car-sales':'cars',
            'paolocons.another-fiat-500-dataset-1538-rows':'cars',
            'antfarol.car-sale-advertisements':'cars',
            ################
            'terminus7.pokemon-challenge':'games',
            'rounakbanik.pokemon':'games',
            'christianlillelund.csgo-round-winner-classification':'games',
            'gdaley.hkracing':'games',
            ################
            'roccoli.gpx-hike-tracks':'other',
            'septa.on-time-performance':'other',
            }

    embeds = pd.read_pickle('embeddings/al_training_plus_11k_datasets_embeddings.pickle')
    ds = reg + cls

    matplotlib.rcParams.update({'font.size': 2})
    plt.rcParams["figure.figsize"] = (2.5,3)

    ######################assign classes###################
    clusters=set(datasets_clusters.values())
    elem_class=[]

    for elem in ds:
        if elem in datasets_clusters:
            elem_class.append(list(clusters).index(datasets_clusters[elem]))
        else:
            elem_class.append(list(clusters).index('other'))

    ########################################################3
    plotted_embeddings = []
    for i in ds:
        if i in embeds['regression'].keys():
            plotted_embeddings.append(embeds['regression'][i]['embedding'])
        elif i in embeds['classification'].keys():
            plotted_embeddings.append(embeds['classification'][i]['embedding'])

    # plt.scatter(features[0], features[1], alpha=0.2,
    #             s=100*features[3], c=iris.target, cmap='viridis')
    # plt.xlabel(iris.feature_names[0])
    # plt.ylabel(iris.feature_names[1]);

    tsne = TSNE(random_state=10, perplexity=5)
    X_tsne = tsne.fit_transform(np.array(plotted_embeddings))
    # print(X_tsne[:, 0], X_tsne[:, 1])
    fig, ax = plt.subplots()

    # cmap = plt.get_cmap('viridis')
    scatt=ax.scatter(X_tsne[:, 0], X_tsne[:, 1],c=elem_class,cmap='tab20')
    for i, txt in enumerate(ds):
        if len(txt.split('.'))> 1:
            ax.annotate(txt.split('.')[1], (X_tsne[:, 0][i], X_tsne[:, 1][i]),ha='center')
        else:
            ax.annotate(txt, (X_tsne[:, 0][i], X_tsne[:, 1][i]-6),ha='center')
    #         ax.annotate(txt, (X_tsne[:, 0][i], X_tsne[:, 1][i]),ha='center')

    legend1 = ax.legend(handles=scatt.legend_elements()[0],
                        loc="center left", labels=list(clusters),prop={'size': 4})
    ax.add_artist(legend1)
    plt.tight_layout()
    # adjust_text(txt, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    plt.savefig('fig.pdf', dpi=1000)
    plt.show()    

if __name__ == '__main__':
    main()