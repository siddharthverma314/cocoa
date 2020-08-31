from systems.neural_system import PytorchNeuralSystem
from neural import model_builder
from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB
from core.scenario import Scenario
from pathlib2 import Path
import pickle
import torch
import json


FILE_PATH = Path(__file__).absolute().parent
SCHEMA_PATH = str(FILE_PATH/"data"/"craigslist-schema.json")
MODEL_PATH = str(FILE_PATH/"checkpoint"/"lf2lf"/"model_best.pt")
PRICE_TRACKER_PATH = str(FILE_PATH/"price_tracker.pkl")
DATA_PATH = str(FILE_PATH/"data"/"dev.json")


def load_neural_system(use_gpu=False):
    args = {
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
        "batch_size": 128,
        "optim": "adagrad",
        "alpha": 0.01,
        "temperature": 0.5,
        "epochs": 30,
        "report_every": 500,
    }
    if use_gpu:
        args.gpuid = 0
    args = type("args", (), args)

    with open(PRICE_TRACKER_PATH) as f:
        price_tracker = pickle.load(f)
    schema = Schema(SCHEMA_PATH)

    system = PytorchNeuralSystem(args, schema, price_tracker, MODEL_PATH, False)
    return system


def get_session_from_uuid(uuid, agent):
    with open(DATA_PATH) as f:
        raw = json.load(f)
    raw = [r["scenario"] for r in raw]
    schema = Schema(SCHEMA_PATH)
    scenario_db = ScenarioDB.from_dict(schema, raw, Scenario)
    kb = scenario_db.get(uuid).get_kb(agent)
    session = system.new_session(agent, kb)
    return session


if __name__ == "__main__":
    system = load_neural_system()
    session = get_session_from_uuid("S_To118PXuNicOd8SO", 0)
