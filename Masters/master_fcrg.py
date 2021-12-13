import sys
sys.path.insert(0, '../SIGIR2019')
sys.path.insert(0, '../../SIGIR2019')

import time
import json
from interactions import MatchInteraction
import matchzoo as mz
from handlers import load_data
import argparse
import random
import numpy as np
import torch
import os
import datetime
from handlers.output_handler import FileHandler
from Models import fcrg_model
from Fitting import basic_fitter


def fit_models(args):
    if not os.path.exists(args.log):
        os.mkdir(args.log)

    curr_date = datetime.datetime.now().timestamp()  # seconds
    # folder to store all outputed files of a run
    secondary_log_folder = os.path.join(args.log, "log_results_%s" % (int(curr_date)))
    if not os.path.exists(secondary_log_folder):
        os.mkdir(secondary_log_folder)

    logfolder_result = os.path.join(secondary_log_folder, "%s_result.txt" % int(curr_date))
    FileHandler.init_log_files(logfolder_result)
    settings = json.dumps(vars(args), sort_keys = True, indent = 2)
    FileHandler.myprint("Running script " + str(os.path.realpath(__file__)))
    FileHandler.myprint(settings)
    FileHandler.myprint("Setting seed to " + str(args.seed))

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    root = args.path
    t1 = time.time()

    train_pack = load_data.load_data2(root, 'train', prefix = args.dataset)
    valid_pack = load_data.load_data2(root, 'dev', prefix = args.dataset)
    predict_pack = load_data.load_data2(root, 'test', prefix = args.dataset)

    a = train_pack.left["text_left"].str.lower().str.split().apply(len).max()
    b = valid_pack.left["text_left"].str.lower().str.split().apply(len).max()
    c = predict_pack.left["text_left"].str.lower().str.split().apply(len).max()
    max_query_length = max([a, b, c])
    min_query_length = min([a, b, c])

    a = train_pack.right["text_right"].str.lower().str.split().apply(len).max()
    b = valid_pack.right["text_right"].str.lower().str.split().apply(len).max()
    c = predict_pack.right["text_right"].str.lower().str.split().apply(len).max()
    max_doc_length = max([a, b, c])
    min_doc_length = min([a, b, c])

    FileHandler.myprint("Min query length, " + str(min_query_length) + " Min doc length " + str(min_doc_length))
    FileHandler.myprint("Max query length, " + str(max_query_length) + " Max doc length " + str(max_doc_length))

    preprocessor = mz.preprocessors.SplitPreprocessor(args.fixed_length_left, args.fixed_length_right,
                                                      vocab_file = os.path.join(args.path, "vocab.json"))
    print('parsing data')
    train_processed = preprocessor.fit_transform(train_pack)  # This is a DataPack
    valid_processed = preprocessor.transform(valid_pack)
    predict_processed = preprocessor.transform(predict_pack)

    train_interactions = MatchInteraction(train_processed)
    valid_interactions = MatchInteraction(valid_processed)
    test_interactions = MatchInteraction(predict_processed)

    FileHandler.myprint('done extracting')
    t2 = time.time()
    FileHandler.myprint('loading data time: %d (seconds)' % (t2 - t1))
    FileHandler.myprint("Building model")

    print("Loading word embeddings......")
    t1_emb = time.time()
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    default_embeddings = mz.datasets.embeddings.load_default_embedding(dimension = args.word_embedding_size,
                                                                       term_index = term_index)
    embedding_matrix = default_embeddings.build_matrix(term_index, initializer=lambda: np.random.normal(0, 1))
    t2_emb = time.time()
    print("Time to load word embeddings......", (t2_emb - t1_emb))

    params = dict()
    params['embedding'] = embedding_matrix
    params["embedding_freeze"] = False  # trainable word embeddings
    params["fixed_length_left"] = args.fixed_length_left
    params["fixed_length_right"] = args.fixed_length_right
    params["embedding_output_dim"] = args.word_embedding_size
    params["embedding_dropout"] = args.embedding_dropout
    params["attention_type"] = args.attention_type
    params["hidden_size"] = args.hidden_size
    params["output_target_size"] = args.output_target_size
    params["bidirectional"] = False
    params["use_label"] = False
    params["use_input_feeding"] = args.use_input_feeding
    params["nlayers"] = 1

    generative_model = fcrg_model.FCRGModel(params)
    FileHandler.myprint("Fitting Model")

    fit_model = basic_fitter.BasicFitter(net=generative_model, loss=args.loss_type,
                                         n_iter=args.epochs, batch_size=args.batch_size,
                                         learning_rate=args.lr, early_stopping=args.early_stopping,
                                         use_cuda=args.cuda, clip=args.clip,
                                         logfolder=secondary_log_folder, curr_date=curr_date,
                                         vocab=preprocessor.context['vocab_unit'])

    try:
        fit_model.fit(train_interactions, verbose = True,
                      val_interactions = valid_interactions,
                      test_interactions = test_interactions)
        fit_model.load_best_model(valid_interactions, test_interactions)

    except KeyboardInterrupt:
        FileHandler.myprint('Exiting from training early')
    t10 = time.time()
    FileHandler.myprint('Total time:  %d (seconds)' % (t10 - t1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Description: Running Neural Text Generation Models")
    parser.add_argument('--path', default = '..', help = 'Input data path', type = str)
    parser.add_argument('--dataset', type = str, default = 'Snopes', help = '[Snopes, Politifact]')
    parser.add_argument('--epochs', default = 100, help = 'Number of epochs to run', type = int)
    parser.add_argument('--batch_size', default = 128, help = 'Batch size', type = int)
    parser.add_argument('--lr', default = 0.001, type = float, help = 'Learning rate')
    parser.add_argument('--early_stopping', default = 10, type = int, help = 'The number of step to stop training')
    parser.add_argument('--log', default = "", type = str, help = 'folder for logs and saved models')
    parser.add_argument('--optimizer', nargs = '?', default = 'adam', help = 'optimizer')
    parser.add_argument('--loss_type', nargs = '?', default = 'cross_entropy',
                        help = 'Specify a loss function: cross entropy or nce')
    parser.add_argument('--word_embedding_size', default = 300, help = 'the dimensions of word embeddings', type = int)
    parser.add_argument('--cuda', type = int, default = 1, help = 'using cuda or not')
    parser.add_argument('--seed', type = int, default = 1111, help = 'random seed')
    parser.add_argument('--fixed_length_left', type = int, default = 89, help = 'Maximum length of each query')
    parser.add_argument('--fixed_length_right', type = int, default = 64, help = 'Maximum length of each document')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--embedding_dropout', type=float, default=0.2,
                        help='dropout applied for embedding (0 means no dropout)')
    parser.add_argument('--attention_type', type = str, default = 'dot', help = 'attention type')
    parser.add_argument('--hidden_size', type = int, default = 300, help = 'hidden size')
    parser.add_argument('--output_target_size', type = int, default = 512, help = 'output_target_size')
    parser.add_argument('--use_input_feeding', type = int, default = 1, help = 'use_input_feeding')

    args = parser.parse_args()
    torch.cuda.set_device(1)
    args.epochs = 100

    args.log = "../logs/fcrg"
    args.dataset = "sigir19"
    args.path = "../formatted_data/politics/mapped_data"
    fit_models(args)
