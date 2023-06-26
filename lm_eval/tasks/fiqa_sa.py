"""
WHEN FLUE MEETS FLANG: Benchmarks and Large Pre-trained Language Model for Financial Domain
https://arxiv.org/pdf/2211.00083.pdf

A financial question answering benchmark for autoregressive language models.

Homepage: https://sites.google.com/view/fiqa/home
"""
from lm_eval.base import Task, rf
from lm_eval.metrics import mean


# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


class FIQASentimentAnalysis(Task):
    VERSION = 0
    DATASET_PATH = "ChanceFocus/fiqa-sentiment-classification"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def _process_doc(self, doc):
        return doc

    def doc_to_text(self, doc):
        return doc["sentence"]

    def doc_to_target(self, doc):
        score = doc["score"]
        if score > 0.05:
            target = "positive"
        elif score < -0.05:
            target = "negative"
        else:
            target = "neutral"
        return " " + target

    def construct_requests(self, doc, ctx):
        ll, is_prediction = rf.loglikelihood(ctx, doc["completion"])
        return is_prediction

    def process_results(self, doc, results):
        (is_prediction,) = results
        return {"acc": is_prediction}

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {"acc": True}
