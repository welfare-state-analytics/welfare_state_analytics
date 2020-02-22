import os
import inspect
import logging

from gensim import models
from gensim.utils import check_output

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_fn_args(f, args):
    return { k: args[k] for k in args.keys()
        if k in inspect.getfullargspec(f).args }

class MalletTopicModel(models.wrappers.LdaMallet):

    def __init__(self, corpus, id2word, default_mallet_home, **args):

        args = filter_fn_args(super(MalletTopicModel, self).__init__, args)

        args.update({ "workers": 4, "optimize_interval": 10 })

        # os.environ["MALLET_HOME"] = default_mallet_home

        mallet_home = os.environ.get('MALLET_HOME', default_mallet_home)

        if not mallet_home:
            raise Exception("Environment variable MALLET_HOME not set. Aborting")

        mallet_path = os.path.join(mallet_home, 'bin', 'mallet') if mallet_home else None

        if os.environ.get('MALLET_HOME', '') != mallet_home:
            os.environ["MALLET_HOME"] = mallet_home

        super(MalletTopicModel, self ).__init__(mallet_path, corpus=corpus, id2word=id2word, **args)

    def ftopicwordweights(self):
        return self.prefix + 'topicwordweights.txt'

    def train(self, corpus):
        self.convert_input(corpus, infer=False)
        cmd = self.mallet_path + ' train-topics --input %s --num-topics %s  --alpha %s --optimize-interval %s '\
            '--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s --topic-word-weights-file %s '\
            '--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s'
        cmd = cmd % (
            self.fcorpusmallet(), self.num_topics, self.alpha, self.optimize_interval,
            self.workers, self.fstate(), self.fdoctopics(), self.ftopickeys(), self.ftopicwordweights(), self.iterations,
            self.finferencer(), self.topic_threshold
        )
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        self.wordtopics = self.word_topics
