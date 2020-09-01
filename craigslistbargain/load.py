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

class Loader:
    """Load the pre-trained policy pickle files from Codalabs. Assumes
    the file directory structure in the Dockerfile.

    """

    SCHEMA_PATH = str(FILE_PATH/"data"/"craigslist-schema.json")
    MODEL_PATH = str(FILE_PATH/"checkpoint"/"lf2lf"/"model_best.pt")
    PRICE_TRACKER_PATH = str(FILE_PATH/"price_tracker.pkl")
    DATA_PATH = str(FILE_PATH/"data"/"dev.json")

    def __init__(self, use_gpu=False):
        # make args that are supposed to be passed in by command line arguments
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

        # HACK: convert args from dict into object. Ex. args["epochs"]
        # becomes args.epochs
        args = type("args", (), args)

        # load price tracker
        with open(self.PRICE_TRACKER_PATH) as f:
            price_tracker = pickle.load(f)

        # load schema
        schema = Schema(self.SCHEMA_PATH)

        # load system
        self.system = PytorchNeuralSystem(
            args, schema, price_tracker, self.MODEL_PATH, False
        )

        # load scenario db
        with open(self.DATA_PATH) as f:
            raw = json.load(f)
        raw = [r["scenario"] for r in raw]  # HACK
        self.scenario_db = ScenarioDB.from_dict(schema, raw, Scenario)

    def from_uuid(self, agent, uuid):
        """Return a session object given a uuid and agent number"""
        scenario = self.scenario_db.get(uuid)
        kb = scenario.get_kb(agent)
        return scenario, self.system.new_session(agent, kb), kb


if __name__ == "__main__":
    from sessions.cmd_session import CmdSession
    from core.controller import Controller
    loader = Loader()
    scenario, session, kb = loader.from_uuid(0, "S_To118PXuNicOd8SO")
    cmd_session = CmdSession(1, kb)
    controller = Controller(scenario, [session, cmd_session])
    controller.simulate()
