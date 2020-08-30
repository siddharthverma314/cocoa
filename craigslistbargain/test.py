from systems.neural_system import PytorchNeuralSystem
from neural import model_builder
from cocoa.core.schema import Schema
import pickle
import torch
import json


if __name__ == "__main__":
    args = type('args', (), {
        "model": "lf2lf",
        "word_vec_size": 300,
        "dropout": 0.,
        "encoder_type": "rnn",
        "decoder_type": "rnn",
        "context_embedder_type": "mean",
        "global_attention": "multibank_general",
        "share_embeddings": False,
        "share_decoder_embeddings": False,
        "enc_layers": 1,
        "copy_attn": False,
        "dec_layers": 1,
        "pretrained_wordvec": "",
        "rnn_size": 300,
        "rnn_type": "LSTM",
        "enc_layers": 1,
        "num_context": 2,
        "stateful": True,
        "sample": True,

        "max_length": 10,
        "n_best": 1,
        # "gpuid": 0,

        "batch_size": 128,
        "optim": "adagrad",
        #"learning_rate": 0.01,
        "alpha": 0.01,
        "temperature": 0.5,
        "epochs": 30,
        "report_every": 500,
        #--cache cache/lf2lf --ignore-cache --verbose --best-only
    })
    schema = Schema("data/craigslist-schema.json")
    model_path = "checkpoint/lf2lf/model_best.pt"
    checkpoint = torch.load(model_path)
    with open("price_tracker.pkl") as f:
        price_tracker = pickle.load(f)
    with open("mappings/lf2lf/vocab.pkl") as f:
        mappings = pickle.load(f)
        mappings['src_vocab'] = mappings['utterance_vocab']
        mappings['tgt_vocab'] = mappings['utterance_vocab']

    #model = model_builder.make_base_model(args, mappings, False, checkpoint=checkpoint)
    system = PytorchNeuralSystem(args, schema, price_tracker, model_path, False)
