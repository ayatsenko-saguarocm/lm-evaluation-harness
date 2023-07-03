"""
Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts
https://arxiv.org/pdf/1307.5336.pdf

The Financial Phrasebank consists of 4,100 sentences drawn from financial news
articles. Each sentence is labeled with one of 12 classes describing the sentiment
of the sentence. The classes are:
    2. Positive
    0. Negative
    1. Neutral

Homepage: N/A
"""
from lm_eval.base import Task, rf
from lm_eval.metrics import mean


_CITATION = """
@article{Malo2014GoodDO,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
  journal={Journal of the Association for Information Science and Technology},
  year={2014},
  volume={65}
}
"""


class FinancialPhrasebank(Task):
    VERSION = 0
    DATASET_PATH = "financial_phrasebank"
    DATASET_NAME = "sentences_50agree"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            print("\n\n\n\\n\n\n\n\n\n\n\\n\n\n\\n\n")
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def test_docs(self):
        if self.has_training_docs():
            print("\n\n\n\\n\n\n\n\n\n\n\\n\n\n\\n\n")
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def doc_to_text(self, doc):
        return doc["sentence"]

    def doc_to_target(self, doc):
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        target = label_map[doc["label"]]
        return " " + target

    def construct_requests(self, doc, ctx):
        ll_positive, _ = rf.loglikelihood(ctx, " positive")
        ll_neutral, _ = rf.loglikelihood(ctx, " neutral")
        ll_negative, _ = rf.loglikelihood(ctx, " negative")

        return ll_positive, ll_neutral, ll_negative

    def process_results(self, doc, results):
        ll_positive, ll_neutral, ll_negative = results

        if ll_positive > ll_neutral and ll_positive > ll_negative:
            pred = " positive"
        elif ll_negative > ll_neutral and ll_negative > ll_positive:
            pred = " negative"
        elif ll_neutral > ll_positive and ll_neutral > ll_negative:
            pred = " neutral"

        target = self.doc_to_target(doc)

        return {"acc": pred == target}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}


class FinancialPhrasebank50(FinancialPhrasebank):
    DATASET_NAME = "sentences_50agree"


class FinancialPhrasebank66(FinancialPhrasebank):
    DATASET_NAME = "sentences_66agree"


class FinancialPhrasebank75(FinancialPhrasebank):
    DATASET_NAME = "sentences_75agree"


class FinancialPhrasebankAll(FinancialPhrasebank):
    DATASET_NAME = "sentences_allagree"
