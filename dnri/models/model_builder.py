from . import encoders
from . import decoders
from . import nri
from . import dnri
import os


def build_model(params):
    if params['model_type'] == 'dnri':
        model = dnri.DNRI(params)
        print("dNRI MODEL: ",model)

    else:
        num_vars = params['num_vars']
        graph_type = params['graph_type']

        # Build Encoder
        encoder = encoders.RefMLPEncoder(params)
        print("ENCODER: ",encoder)

        # Build Decoder
        decoder = decoders.GraphRNNDecoder(params)
        print("DECODER: ",decoder)
        if graph_type == 'dynamic':
            model = nri.DynamicNRI(num_vars, encoder, decoder, params)
        else:
            model = nri.StaticNRI(num_vars, encoder, decoder, params)

    if params['load_best_model']:
        print("LOADING BEST MODEL")
        path = os.path.join(params['working_dir'], 'best_model')
        model.load(path)
    elif params['load_model']:
        print("LOADING MODEL FROM SPECIFIED PATH")
        model.load(params['load_model'])
    if params['gpu']:
        model.cuda()
    return model

