# import os
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
from itertools import count, repeat

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model


def load_model(path):
    sp_model = spm.SentencePieceProcessor(model_file=path)
    proto = sp_pb2_model.ModelProto()
    proto.ParseFromString(sp_model.serialized_model_proto())
    return sp_model, proto


def create_piece(string, score=0):
    new_piece = sp_pb2_model.ModelProto().SentencePiece()
    new_piece.piece = string
    new_piece.score = score
    return new_piece


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default=None, type=str)  # llama
    parser.add_argument("--new", nargs="+")
    parser.add_argument("--model_prefix", required=True)
    parser.add_argument("--scoring", default="bpe", choices=["bpe", "none"])
    args = parser.parse_args()

    # load each model
    original_model, original_proto = load_model(args.original)
    new_models = [load_model(n) for n in args.new]

    if args.scoring == "bpe":
        get_score = count(original_proto.pieces[-1].score - 1, -1)
    else:
        get_score = repeat(0)
    for new_model, new_proto in new_models:
        original_vocab = {p.piece for p in original_proto.pieces}
        for p in new_proto.pieces:
            # do some stuff
            if p.piece not in original_vocab:
                # todo: set scores sensibly
                new_piece = create_piece(p.piece, next(get_score))
                original_proto.pieces.append(new_piece)

    with open(args.model_prefix + ".model", "wb") as f:
        f.write(original_proto.SerializeToString())

    with open(args.model_prefix + ".vocab", "w") as f:
        for piece in original_proto.pieces:
            f.write("\t".join([piece.piece, str(piece.score)]) + "\n")
