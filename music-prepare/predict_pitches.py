import os
os.environ["THEANO_FLAGS"] = "nvcc.flags=-D_FORCE_INLINES"

from blocks.extensions.saveload import load
from theano import function
import numpy
from dataset import T_H5PYDataset
from datetime import datetime
import mxml_utils


def gen():
    source_path = 'dataset/normalized_syllables_rhythm_notes.json-seqlen-100.hdf5'

    train_dataset = T_H5PYDataset(source_path, which_sets=('train',))

    main_loop = load('./checkpoint-pitches.zip')

    model = main_loop.model

    print [x.name for x in model.shared_variables]

    print [x.name for x in model.variables]

    tensor_initial = [x for x in model.shared_variables if x.name == "initial_state"][0]
    tensor_hidden_states = [x for x in model.intermediary_variables if x.name == "hidden_apply_states"][0]
    tensor_x = [x for x in model.variables if x.name == "durations"][0]
    tensor_y = [x for x in model.variables if x.name == "ndim_softmax_apply_output"][0]

    predict_fun = function([tensor_x], tensor_y, updates=[
        (tensor_initial, tensor_hidden_states[0][0]),
    ])

    input_syllables = "love-looks-not-with-the-eyes-but-with-the-mind-and-there-fore-is-winged-cupid-pain-ted-blind"

    from predict import gen

    input_durations = gen(train_dataset, input_syllables)

    input_str_split = input_syllables.split("-")

    input_str_arr = train_dataset.syllables_encode(input_str_split)

    input_str_arr_decoded = train_dataset.syllables_decode(input_str_arr)

    input_durations_arr = train_dataset.durations_encode(input_durations)

    print input_str_arr
    print input_durations_arr

    predictions = []
    import time
    numpy.random.seed(int(time.time()))
    for i in range(len(input_durations_arr)):
        input_char = numpy.zeros((1, 1), dtype=numpy.int32)
        input_char[0][0] = input_durations_arr[i]
        predictions.append(numpy.random.choice(train_dataset.pitches_vocab_size(), 1, p=predict_fun(input_char)[0])[0])

    pitches = train_dataset.pitches_decode(predictions)
    print "Predict:"
    print pitches

    return mxml_utils.create_mxml(input_str_split, input_str_arr_decoded, input_durations, pitches)


if __name__ == "__main__":
    output = gen()
    f = open('generated/rhythm/sample ' + str(datetime.utcnow()) + ".xml", 'w')
    f.write(output)
    f.close()