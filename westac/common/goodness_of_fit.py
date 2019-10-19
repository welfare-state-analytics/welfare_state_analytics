
import scipy

import westac.common.corpus_vectorizer as corpus_vectorizer

class GoodnessOfFit():

    def __init__(self):
        pass

    def fit(self, ytm, method='chisquare', model='uniform'):
        """ Computes how well given distribution fits the year-term-matrix 'ytm' fit

        A goodness-of-fit is computed
        Args:
            ytm (matrix): year-to-term matrix i.e. word distributions over years
            method (str): method to use (not used for now, always `chisquare`)
            model  (str): the model distribution
        """

        corpus = self.create_corpus()
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        vectorizer.fit_transform(corpus)

        id2token = { i: w for w, i in vectorizer.token2id.items() }

        Y = vectorizer.collapse_to_year()
        Yn = vectorizer.normalize(Y, axis=1, norm='l1')

        indices = vectorizer.token_ids_above_threshold(1)
        Ynw = Yn[:, indices]

        X2 = scipy.stats.chisquare(Ynw, f_exp=None, ddof=0, axis=0) # pylint: disable=unused-variable

        # Use X2 so select top 500 words... (highest Power-Power_divergenceResult)
        # Ynw = largest_by_chisquare()
        #print(Ynw)

        linked = linkage(Ynw.T, 'ward') # pylint: disable=unused-variable
        #print(linked)

        labels = [ id2token[x] for x in indices ] # pylint: disable=unused-variable

        #plt.figure(figsize=(24, 16))
        #dendrogram(linked, orientation='top', labels=labels, distance_sort='descending', show_leaf_counts=True)
        #plt.show()
