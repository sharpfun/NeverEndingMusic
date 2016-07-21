__author__ = 'Steffen'

from blocks.bricks.recurrent import LSTM, RecurrentStack, SimpleRecurrent
from blocks.bricks import Tanh, Linear, NDimensionalSoftmax
from blocks import initialization
from blocks.initialization import Constant
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Merge
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.algorithms import GradientDescent, StepClipping, CompositeRule, Adam
from blocks.main_loop import MainLoop
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.saveload import Checkpoint, load
from fuel.datasets.hdf5 import H5PYDataset
from blocks.extensions.monitoring import TrainingDataMonitoring

from blocks_extras.extensions.plot import Plot


import h5py
import theano
from blocks.bricks.cost import CategoricalCrossEntropy
import numpy

from theano import tensor

from bokeh.plotting import Session


class MusicRNNModel:

    def __init__(self, training_data_file):

        self.TrainingDataFile = training_data_file
        with h5py.File(training_data_file) as f:
            self.DurationsVocabSize = len(f.attrs['durations_vocab'])
            self.SyllablesVocabSize = len(f.attrs['syllables_vocab'])
            self.PitchesVocabSize = len(f.attrs['pitches_vocab'])

        self.Cost = None
        self.ComputationGraph = None
        self.Function = None
        self.PitchMainLoop = None
        self.PitchModel = None

    def initialize_pitch_model(self, lookup1_dim=200, lookup2_dim=200, hidden_size=512):

        input_durations = tensor.lmatrix('durations')
        input_syllables = tensor.lmatrix('syllables')
        y = tensor.lmatrix('pitches')

        lookup1 = LookupTable(dim=lookup1_dim, length=self.DurationsVocabSize, name='lookup1',
                              weights_init=initialization.Uniform(width=0.01),
                              biases_init=Constant(0))
        lookup1.initialize()
        lookup2 = LookupTable(dim=lookup2_dim, length=self.SyllablesVocabSize, name='lookup2',
                              weights_init=initialization.Uniform(width=0.01),
                              biases_init=Constant(0))
        lookup2.initialize()
        merge = Merge(['lookup1', 'lookup2'], [lookup1_dim, lookup2_dim], hidden_size,
                              weights_init=initialization.Uniform(width=0.01),
                              biases_init=Constant(0))
        merge.initialize()
        recurrent_block = SimpleRecurrent(dim=hidden_size, activation=Tanh(),
                              weights_init=initialization.Uniform(width=0.01)) #RecurrentStack([LSTM(dim=self.hidden_size, activation=Tanh())] * 3)
        recurrent_block.name = 'recurrent'
        recurrent_block.initialize()
        linear = Linear(input_dim=hidden_size, output_dim=self.PitchesVocabSize,
                              weights_init=initialization.Uniform(width=0.01),
                              biases_init=Constant(0))
        linear.initialize()
        softmax = NDimensionalSoftmax(name='softmax')

        l1 = lookup1.apply(input_durations)
        l2 = lookup2.apply(input_syllables)
        m = merge.apply(l1, l2)
        h = recurrent_block.apply(m)
        a = linear.apply(h)

        self.Cost = softmax.categorical_cross_entropy(y, a, extra_ndim=1).mean()
        self.Cost.name = 'cost'

        y_hat = softmax.apply(a, extra_ndim=1)
        y_hat.name = 'y_hat'

        self.ComputationGraph = ComputationGraph(self.Cost)

        self.Function = None
        self.PitchMainLoop = None
        self.PitchModel = Model(y_hat)

    def train_pitch(self):

        training_data = H5PYDataset(self.TrainingDataFile, which_sets=('train',))

        session = Session(root_url='http://localhost:5006')
        session.login('castle', 'password')

        if self.PitchMainLoop is None:
            step_rules = [Adam()]

            algorithm = GradientDescent(cost=self.Cost,
                                        parameters=self.ComputationGraph.parameters,
                                        step_rule=CompositeRule(step_rules),
                                        on_unused_sources='ignore')

            train_stream = DataStream.default_stream(
                training_data, iteration_scheme=SequentialScheme(
                    training_data.num_examples, batch_size=100))

            self.PitchMainLoop = MainLoop(
                model=Model(self.Cost),
                data_stream=train_stream,
                algorithm=algorithm,
                extensions=[
                    FinishAfter(after_n_epochs=195),
                    Printing(),
                    Checkpoint('trainingdata_pitches.tar', every_n_epochs=10),
                    TrainingDataMonitoring([self.Cost], after_batch=True, prefix='train'),
                    Plot('Pitch generation', channels=[['train_cost']])
                ])

        self.PitchMainLoop.run()

    def sample_pitch(self, input_syllables, input_rhythm):

        output = []

        for (syllable, rhythm) in zip(input_syllables, input_rhythm):
            dist = numpy.exp(self.Function([[syllable]], [[rhythm]])[0])
            output.append(numpy.random.choice(self.PitchesVocabSize, 1, p=dist)[0])

        return output

    def get_var_from(self, name, vars):
        return vars[map(lambda x: x.name, vars).index(name)]

    def load(self):
        self.PitchMainLoop = load(open('trainingdata_pitches.tar'))
        self.PitchModel = self.PitchMainLoop.model

        model_durations = self.get_var_from('durations', self.PitchModel.variables)
        model_syllables = self.get_var_from('syllables', self.PitchModel.variables)
        model_softmax = self.get_var_from('softmax_log_probabilities_output', self.PitchModel.variables)
        model_initial_state = self.get_var_from('initial_state', self.PitchModel.shared_variables)
        model_intermediary_states = self.get_var_from('recurrent_apply_states', self.PitchModel.intermediary_variables)

        self.Function = theano.function([model_durations, model_syllables], model_softmax,
                                        updates=[(model_initial_state, model_intermediary_states[0][0])])
