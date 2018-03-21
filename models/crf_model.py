import json
import logging
import os
import random

import numpy as np

from utils.ConfusionMatrix import ConfusionMatrix
from utils.Progbar import Progbar
from utils.minibatches import minibatches2
from utils.parser_utils import get_chunks

logger = logging.getLogger("NERproject")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class CRFModel(object):
    """
    Wraps any NERModel with a CRF. The CRF does not have an impact on the training. However, after each epoch,
    when predicting on train and dev data, it uses the input of the CRF to tweak the prediction, which is reflected
    on the entity-level scores.
    """

    def __init__(self, model, CRF, args):
        self.model = model
        self.CRF = CRF
        self.alphasCRF = np.concatenate([[0], np.logspace(-2.5, 0, 20)])
        self.alphasOutputPaths = {alpha: self.model.config.output_path + "alpha_" + str(alpha_i) + "/" for
                                  alpha_i, alpha in enumerate(self.alphasCRF)}

        if not self.model.config.no_result_storage:
            for alpha_i, alpha in enumerate(self.alphasCRF):
                os.mkdir(self.model.config.output_path + "alpha_" + str(alpha_i) + "/")
                with open(os.path.join(self.alphasOutputPaths[alpha], "params.json"), "w") as f:
                    dico = vars(args)
                    dico["alpha"] = alpha
                    dico["alpha_i"] = alpha_i
                    json.dump(dico, f, indent=True)

    def outputCRF(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """

        preds = {alpha: [] for alpha in self.alphasCRF}

        prog = Progbar(target=1 + int(len(inputs) / self.model.config.batch_size))
        for i, batch in enumerate(minibatches2(inputs, self.model.config.batch_size, shuffle=False)):
            preds_proba_ = self.model.predict_proba_on_batch(sess, input_batch=batch[0], mask_batch=batch[2])
            preds_ = {alpha: [] for alpha in self.alphasCRF}
            for sentence_preds_proba in preds_proba_:
                for alpha in self.alphasCRF:
                    sentence_preds_ = [np.argmax(sentence_preds_proba[0, :])]
                    for t, pred_proba in enumerate(sentence_preds_proba[1:, :]):
                        previous_predicted_label = sentence_preds_[t - 1]
                        crf_probas = self.CRF.predict_proba_index(previous_predicted_label)
                        combined_probas = alpha * crf_probas + pred_proba
                        sentence_preds_.append(np.argmax(combined_probas))
                    preds_[alpha].append(sentence_preds_)
            for alpha in self.alphasCRF:
                preds[alpha] += list(preds_[alpha])
            prog.update(i + 1, [])

        return {alpha: self.model.consolidate_predictions(inputs_raw, inputs, preds[alpha]) for alpha in self.alphasCRF}

    def evaluateCRF(self, sess, examples, examples_raw, evaluate=True):
        if not evaluate:
            return {alpha: (0., 0., 0.) for alpha in self.alphasCRF}

        token_cm = ConfusionMatrix(labels=self.model.labelsHandler.keys())

        correct_preds, total_correct, total_preds = 0., 0., 0.
        outputsByAlpha = self.outputCRF(sess, examples_raw, examples)
        scoresByAlpha = {}
        for alpha in outputsByAlpha:
            output = outputsByAlpha[alpha]
            for _, labels, labels_ in output:
                for l, l_ in zip(labels, labels_):
                    token_cm.update(l, l_)
                gold = set(get_chunks(labels, self.model.labelsHandler.noneIndex()))
                pred = set(get_chunks(labels_, self.model.labelsHandler.noneIndex()))
                correct_preds += len(gold.intersection(pred))
                total_preds += len(pred)
                total_correct += len(gold)

            p = correct_preds / total_preds if correct_preds > 0 else 0
            r = correct_preds / total_correct if correct_preds > 0 else 0
            f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
            scoresByAlpha[alpha] = (p, r, f1)
        return scoresByAlpha

    def fit(self, sess, saver, train_examples_raw, dev_examples_raw):
        best_score = 0.

        train_examples = self.model.preprocess_sequence_data(train_examples_raw)
        dev_examples = self.model.preprocess_sequence_data(dev_examples_raw)

        learning_rate_decay = 1.
        for epoch in range(self.model.config.n_epochs):
            if epoch > self.model.config.n_epochs / 2:
                learning_rate_decay = 1. / 3.
            if epoch > self.model.config.n_epochs * 3 / 4:
                learning_rate_decay = 1. / 9.
            logger.info("Epoch %d out of %d", epoch + 1, self.model.config.n_epochs)

            prog = Progbar(target=1 + int(len(train_examples) / self.model.config.batch_size))

            losses = []
            for i, minibatch in enumerate(
                    minibatches2(train_examples, self.model.config.batch_size, shuffle=True)):
                loss = self.model.train_on_batch(sess, *minibatch, learning_rate_decay=learning_rate_decay)
                losses.append(loss)
                prog.update(i + 1, [("loss = ", loss)])

            random_train_examples_id = random.sample(range(len(train_examples)), k=len(dev_examples))
            logger.info("Evaluating on training data")
            try:
                train_entity_scoresByAlpha = self.evaluateCRF(sess,
                                                              [train_examples[ind] for ind in random_train_examples_id],
                                                              [train_examples_raw[ind] for ind in
                                                               random_train_examples_id],
                                                              evaluate=(epoch == self.model.config.n_epochs - 1))
            except IndexError:
                train_entity_scoresByAlpha = {alpha: (-1, -1, -1) for alpha in self.alphasCRF}

            for alpha in train_entity_scoresByAlpha:
                train_entity_scores = train_entity_scoresByAlpha[alpha]
                p, r, f1 = train_entity_scores
                if not self.model.config.no_result_storage:
                    with open(self.alphasOutputPaths[alpha] + "train_losses.los", "a") as f:
                        for item in losses:
                            f.write("%s\n" % item)
                    with open(self.alphasOutputPaths[alpha] + "train_f1.los", "a") as f:
                        f.write("%s\n" % f1)
                    with open(self.alphasOutputPaths[alpha] + "train_precision.los", "a") as f:
                        f.write("%s\n" % p)
                    with open(self.alphasOutputPaths[alpha] + "train_recall.los", "a") as f:
                        f.write("%s\n" % r)

            logger.info("Evaluating on development data")
            for dev_minibatch in minibatches2(dev_examples, len(dev_examples)):
                dev_loss = self.model.test_on_batch(sess, *dev_minibatch)[0]
                break
            dev_entity_scoresByAlpha = self.evaluateCRF(sess,
                                                        dev_examples,
                                                        dev_examples_raw,
                                                        evaluate=(epoch == self.model.config.n_epochs - 1))

            bestAlphaF1 = max(self.alphasCRF, key=lambda alpha: dev_entity_scoresByAlpha[alpha][-1])
            logger.info("Entity level best P/R/F1: %.2f/%.2f/%.2f", *(dev_entity_scoresByAlpha[bestAlphaF1]))
            logger.info("Reached for alpha value %.2f", bestAlphaF1)
            logger.info("Dev set loss: %.4f", dev_loss)
            logger.info("For job at " + self.model.config.output_path)

            for alpha in dev_entity_scoresByAlpha:
                dev_entity_scores = dev_entity_scoresByAlpha[alpha]
                p, r, f1 = dev_entity_scores
                if not self.model.config.no_result_storage:
                    with open(self.alphasOutputPaths[alpha] + "dev_losses.los", "a") as f:
                        f.write("%s\n" % dev_loss)
                    with open(self.alphasOutputPaths[alpha] + "dev_f1.los", "a") as f:
                        f.write("%s\n" % f1)
                    with open(self.alphasOutputPaths[alpha] + "dev_precision.los", "a") as f:
                        f.write("%s\n" % p)
                    with open(self.alphasOutputPaths[alpha] + "dev_recall.los", "a") as f:
                        f.write("%s\n" % r)

        return best_score
