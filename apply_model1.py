

from datetime import datetime
from os import path
import argparse
import pickle
import numpy as np
import sys
import tensorflow as tf

sys.path.append(path.join("..", "src"))

from tflego import NP_DTYPE, TF_DTYPE, NP_ITYPE, TF_ITYPE
import batching
import data_io


#-----------------------------------------------------------------------------#
#                            APPLY MODEL FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def build_model(x, x_lengths, options_dict):
    model_dict = {}
    if options_dict["script"] == "train_cae":
        import train_cae_final
        cae = train_cae_final.build_cae_from_options_dict(
            x, x_lengths, x_lengths, options_dict
            )
        model_dict["output"] = cae["y"]
        model_dict["encoding"] = cae["z"]
        model_dict["mask"] = cae["mask"]
    if options_dict["script"] == "train_cvae":
        import train_cvae_final
        cvae = train_cvae_final.build_cae_from_options_dict(
            x, x_lengths, x_lengths, options_dict
            )
        model_dict["output"] = cvae["y"]
        model_dict["encoding"] = cvae["z_mean"]
        model_dict["mask"] = cvae["mask"]

    elif options_dict["script"] == "train_vae":
        import train_vae_final
        vae = train_vae_final.build_vae_from_options_dict(x, x_lengths, options_dict)
        model_dict["output"] = vae["decoder_output"]
        model_dict["encoding"] = vae["latent_layer"]["z_mean"]
        model_dict["mask"] = vae["mask"]
    elif options_dict["script"] == "train_siamese":
        import train_siamese
        siamese = train_siamese.build_siamese_from_options_dict(
            x, x_lengths, options_dict
            )
        model_dict["encoding"] = siamese["output"]
    elif options_dict["script"] == "train_siamese_cnn":
        import train_siamese_cnn
        siamese = train_siamese_cnn.build_siamese_cnn_from_options_dict(
            x, options_dict
            )
        model_dict["encoding"] = siamese["output"]
    return model_dict


# def apply_model(model_fn, subset, language, batch_size=None):
def apply_model(model_fn, subset, batch_size=None):

    # assert language is None  # to-do

    # Load the model options
    model_dir = path.split(model_fn)[0]
    options_dict_fn = path.join(model_dir, "options_dict.pkl")
    print("Reading:", options_dict_fn)
    with open(options_dict_fn, "rb") as f:
        options_dict = pickle.load(f)

    # Load data
    npz_fn = path.join(options_dict["data_dir"], subset + ".npz")
    # if language is not None:
    #     if "buckeye" in npz_fn:
    #         npz_fn = npz_fn.replace("buckeye", language)
    #     elif "xitsonga" in npz_fn:
    #         npz_fn = npz_fn.replace("xitsonga", language)
    x_data, labels, lengths, keys= data_io.load_data_from_npz(
        npz_fn
        )

    if "cnn" in options_dict["script"]:

        # Pad and flatten data
        x_data, _ = data_io.pad_sequences(
            x_data, options_dict["max_length"], True
            )
        x_data = np.transpose(x_data, (0, 2, 1))
        x_data = x_data.reshape((-1, options_dict["d_in"]))

        # Build model
        x = tf.placeholder(TF_DTYPE, [None, options_dict["d_in"]])
        model = build_model(x, None, options_dict)

        # Embed data
        if batch_size is None:
            batch_iterator = batching.LabelledIterator(
                x_data, None, x_data.shape[0], False
                )
            saver = tf.train.Saver()
            with tf.Session() as session:
                saver.restore(session, model_fn)
                for batch_x in batch_iterator:
                    np_z = session.run(
                        [model["encoding"]], feed_dict={x: batch_x})[0]
                    break  # single batch
        else:
            assert False, "need to implement"

    else:  # rnn
        
        # Truncate and limit dimensionality
        data_io.trunc_and_limit_dim(
            x_data, lengths, options_dict["n_input"], options_dict["max_length"]
            )

        # Build model
        x = tf.placeholder(TF_DTYPE, [None, None, options_dict["n_input"]])
        x_lengths = tf.placeholder(TF_ITYPE, [None])
        model = build_model(x, x_lengths, options_dict)

        # Embed data
        if batch_size is None:
            batch_iterator = batching.SimpleIterator(
                x_data, len(x_data), False
                )
            saver = tf.train.Saver()
            with tf.Session() as session:
                saver.restore(session, model_fn)
                for batch_x_padded, batch_x_lengths in batch_iterator:
                    np_x = batch_x_padded
                    np_x_lengths = batch_x_lengths
                    np_z = session.run(
                        [model["encoding"]], feed_dict={x: np_x, x_lengths:
                        np_x_lengths}
                        )[0]
                    break  # single batch
        else:
            batch_iterator = batching.SimpleIterator(
                x_data, batch_size, False
                )
            saver = tf.train.Saver()
            with tf.Session() as session:
                saver.restore(session, model_fn)
                np_z = []
                for batch_x_padded, batch_x_lengths in batch_iterator:
                    np_x = batch_x_padded
                    np_x_lengths = batch_x_lengths
                    cur_np_z = session.run(
                        [model["encoding"]], feed_dict={x: np_x, x_lengths:
                        np_x_lengths}
                        )[0]
                    print("!", cur_np_z.shape)
                    np_z.append(cur_np_z)
                np_z = np.vstack(np_z)
                print("!", np_z.shape)

    embed_dict = {}
    for i, utt_key in enumerate([keys[i] for i in batch_iterator.indices]):
        embed_dict[utt_key] = np_z[i]

    return embed_dict


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("model_fn", type=str, help="model checkpoint filename")
    parser.add_argument(
        "subset", type=str, help="subset to perform evaluation on",
        choices=["train.gt", "val", "test"]
        )
    parser.add_argument(
        "--language", type=str,
        help="if provided, this language is used instead of the language in "
        "the model's options dictionary"
        )
    parser.add_argument(
        "--batch_size", type=int, 
        help="size of mini-batch (default: %(default)s)",
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Embed data
    embed_dict = apply_model(
        args.model_fn, args.subset, args.batch_size
        )

    # Save embeddings
    model_dir, model_fn = path.split(args.model_fn)
    npz_fn =  args.subset + ".npz" 
    npz_fn = path.join(model_dir, path.splitext(model_fn)[0] + "." + npz_fn)
    print("Writing:", npz_fn)
    np.savez_compressed(npz_fn, **embed_dict)
    print(datetime.now())


if __name__ == "__main__":
    main()
