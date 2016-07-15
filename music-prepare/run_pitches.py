import os
os.environ["THEANO_FLAGS"] = "nvcc.flags=-D_FORCE_INLINES"

from theano import tensor
from blocks import initialization
from blocks.initialization import Constant
from blocks.bricks import Tanh
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.bricks import Linear, Softmax, NDimensionalSoftmax
import json
from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp
from dataset import T_H5PYDataset


source_path = 'dataset/normalized_syllables_rhythm_notes.json-seqlen-100.hdf5'


train_dataset = T_H5PYDataset(source_path, which_sets=('train',))


hidden_layer_dim = 200

x = tensor.lmatrix('durations')
y = tensor.lmatrix('pitches')

lookup_input = LookupTable(
    name='lookup_input',
    length=train_dataset.durations_vocab_size()+1,
    dim=hidden_layer_dim,
    weights_init=initialization.Uniform(width=0.01),
    biases_init=Constant(0))
lookup_input.initialize()

linear_input = Linear(
    name='linear_input',
    input_dim=hidden_layer_dim,
    output_dim=hidden_layer_dim,
    weights_init=initialization.Uniform(width=0.01),
    biases_init=Constant(0))
linear_input.initialize()

rnn = SimpleRecurrent(
    name='hidden',
    dim=hidden_layer_dim,
    activation=Tanh(),
    weights_init=initialization.Uniform(width=0.01))
rnn.initialize()

linear_output = Linear(
    name='linear_output',
    input_dim=hidden_layer_dim,
    output_dim=train_dataset.pitches_vocab_size(),
    weights_init=initialization.Uniform(width=0.01),
    biases_init=Constant(0))
linear_output.initialize()

softmax = NDimensionalSoftmax(name='ndim_softmax')

activation_input = lookup_input.apply(x)
hidden = rnn.apply(linear_input.apply(activation_input))
activation_output = linear_output.apply(hidden)
y_est = softmax.apply(activation_output, extra_ndim=1)

cost = softmax.categorical_cross_entropy(y, activation_output, extra_ndim=1).mean()


from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Adam

cg = ComputationGraph([cost])

step_rules = [RMSProp(learning_rate=0.002, decay_rate=0.95), StepClipping(1.0)]


algorithm = GradientDescent(
    cost=cost,
    parameters=cg.parameters,
    step_rule=CompositeRule(step_rules),
    on_unused_sources='ignore'
)


from blocks.extensions import Timing, FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.main_loop import MainLoop
from blocks.extensions.saveload import Checkpoint


from blocks.model import Model

main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=DataStream.default_stream(
        dataset=train_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_examples, batch_size=20)
    ),
    model=Model(y_est),
    extensions=[
        Timing(),
        FinishAfter(after_n_epochs=200),
        TrainingDataMonitoring(
            variables=[cost],
            prefix="train",
            after_epoch=True
        ),
        Printing(),
        ProgressBar(),
        Checkpoint(path="./checkpoint-pitches.zip")
    ]
)

main_loop.run()


