import torch
import numpy as np
import torch_utils
from Models import base_model
import torch.nn.functional as F
import torch_utils as my_utils
import torch.optim as optim
import time
import os
from handlers import output_handler, mz_sampler
import json
import matchzoo
import interactions
from handlers.output_handler import FileHandler
from handlers.tensorboard_writer import TensorboardWrapper
from setting_keywords import KeyWordSettings


class BasicFitter(object):

    def __init__(self, net: base_model.BaseModel,
                 loss = "cross_entropy",
                 n_iter = 100,
                 testing_epochs = 5,
                 batch_size = 16,
                 learning_rate = 1e-4,
                 early_stopping = 0,  # means no early stopping
                 clip = None,
                 optimizer_func = None,
                 use_cuda = False,
                 logfolder = None,
                 curr_date = None,
                 **kargs):
        self._loss = loss
        self._n_iter = n_iter
        self._testing_epochs = testing_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._clip = clip
        self._optimizer_func = optimizer_func
        self._use_cuda = use_cuda
        self._early_stopping_patience = early_stopping # for early stopping

        self._net = net
        self._optimizer = None
        assert logfolder != ""
        self.logfolder = logfolder
        if not os.path.exists(logfolder):
            os.mkdir(logfolder)

        self.saved_model = os.path.join(logfolder, "%s_saved_model" % int(curr_date))
        TensorboardWrapper.init_log_files(os.path.join(logfolder, "tensorboard_%s" % int(curr_date)))
        # for evaluation during training
        self._sampler = mz_sampler.Sampler()
        self._candidate = dict()
        self._vocab = kargs[KeyWordSettings.Vocab]
        self.index_of_pad_token = self._vocab._state['term_index'][self._vocab._pad]

    def _initialize(self):
        # put the model into cuda if use cuda
        self._net = my_utils.gpu(self._net, self._use_cuda)
        self._optimizer = optim.Adam(self._net.parameters(), lr=self._learning_rate)

    def fit(self, train_iteractions: interactions.MatchInteraction,
            verbose = True,  # for printing out evaluation during training
            val_interactions: interactions.MatchInteraction = None,
            test_interactions: interactions.MatchInteraction = None):
        """
        Fit the model.
        Parameters
        ----------
        train_iteractions: :class:`matchzoo.DataPack` The input sequence dataset.
        val_interactions: :class:`matchzoo.DataPack`
        test_interactions: :class:`matchzoo.DataPack`
        """
        self._initialize()
        best_ce, best_epoch, test_ce = 0, 0, 0
        test_results_dict = None
        iteration_counter = 0
        count_patience_epochs = 0

        for epoch_num in range(self._n_iter):
            # ------ Move to here ----------------------------------- #
            self._net.train(True)
            query_ids, left_contents, left_lengths, \
            doc_ids, right_contents, target_contents, right_lengths = self._sampler.get_instances(train_iteractions)

            queries, query_content, query_lengths, \
            docs, doc_content, target_contents, doc_lengths = my_utils.shuffle(query_ids, left_contents, left_lengths,
                                                              doc_ids, right_contents, target_contents, right_lengths)
            epoch_loss, total_pairs = 0.0, 0
            t1 = time.time()
            for (minibatch_num, (batch_query, batch_query_content, batch_query_len,
                 batch_doc, batch_doc_content, batch_doc_target, batch_docs_lens)) \
                    in enumerate(my_utils.minibatch(queries, query_content, query_lengths,
                                                    docs, doc_content, target_contents, doc_lengths,
                                                    batch_size = self._batch_size)):
                t3 = time.time()
                batch_query = my_utils.gpu(torch.from_numpy(batch_query), self._use_cuda)
                batch_query_content = my_utils.gpu(torch.from_numpy(batch_query_content), self._use_cuda)
                # batch_query_len = my_utils.gpu(torch.from_numpy(batch_query_len), self._use_cuda)
                batch_doc = my_utils.gpu(torch.from_numpy(batch_doc), self._use_cuda)
                batch_doc_content = my_utils.gpu(torch.from_numpy(batch_doc_content), self._use_cuda)
                batch_doc_target = my_utils.gpu(torch.from_numpy(batch_doc_target), self._use_cuda)
                # batch_docs_lens = my_utils.gpu(torch.from_numpy(batch_docs_lens), self._use_cuda)

                total_pairs += batch_query.size(0) # (batch_size)
                self._optimizer.zero_grad()
                loss = self._get_loss(batch_query, batch_query_content,
                                      batch_doc, batch_doc_content, batch_query_len, batch_docs_lens,
                                      batch_doc_target)
                epoch_loss += loss.item()
                iteration_counter += 1
                # if iteration_counter % 2 == 0: break
                TensorboardWrapper.mywriter().add_scalar("loss/minibatch_loss", loss.item(), iteration_counter)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), self._clip)
                self._optimizer.step()
                t4 = time.time()
                if iteration_counter % 100 == 0: print("Running time for each mini-batch: ", (t4 - t3), "s")
            epoch_loss /= float(total_pairs)
            TensorboardWrapper.mywriter().add_scalar("loss/epoch_loss_avg", epoch_loss, epoch_num)
            # print("Number of Minibatches: ", minibatch_num, "Avg. loss of epoch: ", epoch_loss)
            t2 = time.time()
            epoch_train_time = t2 - t1
            if verbose:  # validation after each epoch
                t1 = time.time()
                result_val = self.evaluate(val_interactions)
                val_ce = result_val["cross_entropy"]
                t2 = time.time()
                validation_time = t2 - t1

                TensorboardWrapper.mywriter().add_scalar("cross_entropy/val_ce", val_ce, epoch_num)
                FileHandler.myprint('|Epoch %03d | Train time: %04.1f(s) | Train loss: %.3f'
                                    '| Val loss = %.5f | Validation time: %04.1f(s)'
                                    % (epoch_num, epoch_train_time, epoch_loss, val_ce, validation_time))

                if val_ce < best_ce:
                    count_patience_epochs = 0
                    with open(self.saved_model, "wb") as f:
                        torch.save(self._net.state_dict(), f)
                    # test_results_dict = result_test
                    best_ce, best_epoch = val_ce, epoch_num
                else: count_patience_epochs += 1
                if self._early_stopping_patience and count_patience_epochs > self._early_stopping_patience:
                    FileHandler.myprint("Early Stopped due to no better performance in %s epochs" % count_patience_epochs)
                    break

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'.format(epoch_loss))
        FileHandler.myprint("Closing tensorboard")
        TensorboardWrapper.mywriter().close()
        FileHandler.myprint('Best result: | vad cross_entropy = %.5f | epoch = %d' % (best_ce, best_epoch))
        FileHandler.myprint_details(json.dumps(test_results_dict, sort_keys = True, indent = 2))

    def _get_loss(self, query_ids: torch.Tensor,
                  query_contents: torch.Tensor,
                  doc_ids: torch.Tensor,
                  doc_contents: torch.Tensor,
                  query_lens: np.ndarray,
                  docs_lens: np.ndarray,
                  target_contents, **kargs) -> torch.Tensor:
        """
        Compute loss for batch_size pairs. Note: Query and Doc have different lengths

        :param query_ids: (B, )
        :param query_contents: (B, L)
        :param doc_ids: (B, )
        :param doc_contents: (B, R)
        :param query_lens: (B, )
        :param docs_lens: (B, )
        :param target_contents: (B, R)
        :param kargs:
        :return:
        """
        batch_size = query_ids.size(0)
        L2 = doc_contents.size(1)
        L1 = query_contents.size(1)

        q_new_indices, q_restoring_indices = torch_utils.get_sorted_index_and_reverse_index(query_lens)
        query_lens = my_utils.gpu(torch.from_numpy(query_lens), self._use_cuda)

        d_new_indices, d_old_indices = torch_utils.get_sorted_index_and_reverse_index(docs_lens)
        docs_lens = my_utils.gpu(torch.from_numpy(docs_lens), self._use_cuda)
        additional_paramters = {
            KeyWordSettings.Query_lens: query_lens,
            KeyWordSettings.QueryLensIndices: (q_new_indices, q_restoring_indices, query_lens),
            KeyWordSettings.Doc_lens: docs_lens,
            KeyWordSettings.DocLensIndices: (d_new_indices, d_old_indices, docs_lens),
            KeyWordSettings.UseCuda: self._use_cuda
        }
        logits = self._net(query_contents, doc_contents, query_lens, None, **additional_paramters)
        num_classes = len(self._vocab._state['term_index'])
        logits = logits.view(-1, num_classes)  # (B * R, C)
        target_contents = target_contents.view(-1)  # (B, R) => (B * R)
        loss = F.cross_entropy(logits, target_contents, ignore_index=self.index_of_pad_token)
        return loss

    def load_best_model(self, val_interactions: interactions.MatchInteraction,
                        test_interactions: interactions.MatchInteraction):
        mymodel = self._net
        mymodel.load_state_dict(torch.load(self.saved_model))
        mymodel.train(False)
        my_utils.gpu(mymodel, self._use_cuda)

        val_results = self.evaluate(val_interactions)
        val_loss = val_results["cross_entropy"]
        test_results = self.evaluate(test_interactions)
        test_loss = test_results["cross_entropy"]

        FileHandler.save_error_analysis_validation(json.dumps(val_results, sort_keys = True, indent = 2))
        FileHandler.save_error_analysis_testing(json.dumps(test_results, sort_keys = True, indent = 2))
        FileHandler.myprint('Best val loss = %.5f |Best Test loss = %.5f ' % (val_loss, test_loss))

        return val_loss, test_loss

    def evaluate(self, testRatings: interactions.MatchInteraction, output_ranking = False, **kargs):
        self._net.train(False)  # disabling training
        query_ids, left_contents, left_lengths, \
        doc_ids, right_contents, target_contents, right_lengths = self._sampler.get_instances(testRatings)
        eval_loss = 0.0
        total_tokens = 0
        for (minibatch_num,
             (batch_query, batch_query_content, batch_query_len,
              batch_doc, batch_doc_content, batch_doc_target, batch_docs_lens)) \
                in enumerate(my_utils.minibatch(query_ids, left_contents, left_lengths,
                                                doc_ids, right_contents, target_contents, right_lengths,
                                                batch_size=self._batch_size)):

            batch_query = my_utils.gpu(torch.from_numpy(batch_query), self._use_cuda)
            batch_query_content = my_utils.gpu(torch.from_numpy(batch_query_content), self._use_cuda)
            # batch_query_len = my_utils.gpu(torch.from_numpy(batch_query_len), self._use_cuda)
            batch_doc = my_utils.gpu(torch.from_numpy(batch_doc), self._use_cuda)
            batch_doc_content = my_utils.gpu(torch.from_numpy(batch_doc_content), self._use_cuda)
            batch_doc_target = my_utils.gpu(torch.from_numpy(batch_doc_target), self._use_cuda)
            # batch_docs_lens = my_utils.gpu(torch.from_numpy(batch_docs_lens), self._use_cuda)
            batch_loss = self._get_loss(batch_query, batch_query_content, batch_doc, batch_doc_content,
                                        batch_query_len, batch_docs_lens, batch_doc_target)
            mask = (batch_doc_target != self.index_of_pad_token)
            non_pad_tokens = torch.sum(mask).float()
            loss = batch_loss.data.cpu().numpy()
            loss *= non_pad_tokens
            eval_loss += loss
            total_tokens += non_pad_tokens

        eval_loss /= total_tokens
        results = dict()
        results["cross_entropy"] = eval_loss
        return results
