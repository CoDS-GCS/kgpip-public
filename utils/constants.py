import pathlib

KGPIP_PATH = str(pathlib.Path(__file__).parent.parent.resolve())


al_evaluation = [('kaggle_datasets', 'housing-prices', True, 'SalePrice', 0.84),
                 ('kaggle_datasets', 'mercedes-benz-greener-manufacturing', True, 'y', 0.52),
                 ('kaggle_datasets', 'detecting-insults-in-social-commentary', False, 'Insult', 0.77),
                 ('kaggle_datasets', 'sentiment-analysis-on-movie-reviews', False, 'Sentiment', 0.33),
                 ('kaggle_datasets', 'titanic', False, 'Survived', 0.81),
                 ('kaggle_datasets', 'spooky-author-identification', False, 'author', 0.84)]


dataset_names_11k_scripts = [
                            'benhamner.2016-us-election',
                            'SIZZLE.2016electionmemes',
                            'wcukierski.2017-march-ml-mania-predictions',
                            'etsc9287.2020-general-election-polls',
                            'jiashenliu.515k-hotel-reviews-data-in-europe',
                            'dylanjcastillo.7k-books-with-metadata',
                            'raghadalharbi.all-products-available-on-sephora-website',
                            'salmanfaroz.amazon-stock-price-1997-to-2020',
                            'alancmathew.anime-dataset',
                            'CooperUnion.anime-recommendations-database',
                            'vishalmane109.anime-recommendations-database',
                            'paolocons.another-fiat-500-dataset-1538-rows',
                            'gmadevs.atp-matches-dataset',
                            'gbonesso.b3-stock-quotes',
                            'barelydedicated.bank-customer-churn-modeling',
                            'jacobbaruch.basketball-players-stats-per-season-49-leagues',
                            'danielkyrka.bmw-pricing-challenge',
                            'andreifnmg.campeonato-braileiro-20092018',
                            'antfarol.car-sale-advertisements',
                            'gagandeep16.car-sales',
                            'epa.carbon-monoxide',
                            'abineshkumark.carsdata',
                            'mathchi.churn-for-bank-customers',
                            'everling.cocaine-listings',
                            'andrewsundberg.college-basketball-dataset',
                            'mariotormo.complete-pokemon-dataset-updated-090420',
                            'unanimad.corona-virus-brazil',
                            'anjanatiha.corona-virus-time-series-dataset',
                            'praveengovi.coronahack-chest-xraydataset',
                            'parulpandey.coronavirus-cases-in-india',
                            'priteshraj10.coronavirus-covid19-drug-discovery',
                            'smid80.coronavirus-covid19-tweets',
                            'smid80.coronavirus-covid19-tweets-early-april',
                            'smid80.coronavirus-covid19-tweets-late-april',
                            'kimjihoo.coronavirusdataset',
                            'transparencyint.corruption-index',
                            'aestheteaman01.covcsd-covid19-countries-statistical-dataset',
                            'atilamadai.covid19',
                            'ridoy11.covid19-bangladesh-dataset',
                            'howsmyflattening.covid19-challenges',
                            'vignesh1694.covid19-coronavirus',
                            'antgoldbloom.covid19-data-from-john-hopkins-university',
                            'lisphilar.covid19-dataset-in-japan',
                            'rohanrao.covid19-forecasting-metadata',
                            'mohitmaithani.covid19-full-dataset',
                            'dgrechka.covid19-global-forecasting-locations-population',
                            'sudalairajkumar.covid19-in-india',
                            'sudalairajkumar.covid19-in-italy',
                            'kimdanny.covid19-in-south-korea',
                            'hendratno.covid19-indonesia',
                            'tanmoyx.covid19-patient-precondition-dataset',
                            'phiitm.covid19-research-preprint-data',
                            'kapral42.covid19-russia-regions-cases',
                            'skylord.covid19-tests-conducted-by-country',
                            'gpreda.covid19-tweets',
                            'lopezbec.covid19-tweets-dataset',
                            'terenceshin.covid19s-impact-on-airport-traffic',
                            'austinreese.craigslist-carstrucks-data',
                            'sakshigoyal7.credit-card-customers',
                            'cclayford.cricinfo-statsguru-data',
                            'sameerkulkarni91.crime-in-ireland',
                            'AnalyzeBoston.crimes-in-boston',
                            'adamschroeder.crimes-new-york-city',
                            'mateusdmachado.csgo-professional-matches',
                            'christianlillelund.csgo-round-winner-classification',
                            'mrmorj.data-police-shootings',
                            'shashwatwork.dataco-smart-supply-chain-for-big-data-analysis',
                            'rymnikski.dataset-for-collaborative-filters',
                            'uciml.default-of-credit-card-clients-dataset',
                            'hgiydan.detroit-crimeincidents20092016',
                            'devinanzelmo.dota-2-matches',
                            'sukanthen.dream11-ipl2020-live',
                            'lucabasa.dutch-energy',
                            'benroshan.ecommerce-data',
                            'shashwatwork.ecommerce-data',
                            'danerbland.electionfinance',
                            'ellipticco.elliptic-data-set',
                            'ringhilterra17.enrichednytimescovid19',
                            'futurecorporation.epitope-prediction',
                            'rushikeshhiray.esport-earnings',
                            'sagunsh.fifa-20-complete-player-dataset',
                            'stefanoleone992.fifa-20-complete-player-dataset',
                            'bryanb.fifa-player-stats-database',
                            'notlucasp.financial-news-headlines',
                            'cjgdev.formula-1-race-data-19502017',
                            'rohanrao.formula-1-world-championship-1950-2020',
                            'atharvap329.glassdoor-data-science-job-data',
                            'ikiulian.global-hospital-beds-capacity-for-covid19',
                            'jealousleopard.goodreadsbooks',
                            'jpallard.google-store-ecommerce-data-fake-retail-data',
                            'roccoli.gpx-hike-tracks',
                            'mchavoshi.grouplens-2018',
                            'reginakhu.happiness',
                            'rohitmathur100.happiness',
                            'sohoooon.happiness',
                            'gdaley.hkracing',
                            'murderaccountability.homicide-reports',
                            'dheerajmpai.hospitals-and-beds-in-india',
                            'fifthtribe.how-isis-uses-twitter',
                            'cbp.illegal-immigrants',
                            'satkarjain.imdb-movie-19722019',
                            'martj42.international-football-results-from-1872-to-2017',
                            'mrisdal.minneapolis-incidents-crime',
                            'asad1m9a9h6mood.news-articles',
                            'septa.on-time-performance',
                            'abcsds.pokemon',
                            'mlomuscio.pokemon',
                            'rounakbanik.pokemon',
                            'shikhar1.pokemon',
                            'yvonhk.pokemon',
                            'terminus7.pokemon-challenge',
                            'nhtsa.safety-recalls',
                            'aaron7sun.stocknews',
                            'jmmvutu.summer-products-and-sales-in-ecommerce-wish',
                            'ali2020armor.taekwondo-techniques-classification',
                            'rajeevw.ufcdata',
                            'sobhanmoosavi.us-accidents',
                            'daithibhard.us-electoral-college-votes-per-state-17882020',
                            'nicapotato.womens-ecommerce-clothing-reviews',
                            'timoboz.womens-ecommerce-clothing-reviews',
                            'danevans.world-bank-wdi-212-health-systems',
                            'unsdsn.world-happiness',
                            'PromptCloudHQ.world-happiness-report-2019',
                            'miroslavsabo.young-people-survey',
                            'nxpnsv.country-health-indicators',
                            'newzoel.covid-19-cssegisanddata',
                            'rohitrox.healthcare-provider-fraud-detection-analysis',
                            'fec.independent-political-ad-spending',
                            'regivm.retailtransactiondata',
                            'adityadesai13.used-car-dataset-ford-and-mercedes',
                            'moezabid.zillow-all-homes-data',
                            'unanimad.us-election-2020',
                            'tunguz.us-elections-dataset'
                            ]
