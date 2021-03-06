__author__ = 'Steffen'

from blocks.bricks.recurrent import LSTM, RecurrentStack, SimpleRecurrent
from blocks.bricks import Tanh, Linear, NDimensionalSoftmax
from blocks import initialization
from blocks.initialization import Constant
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Merge
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp
from blocks.main_loop import MainLoop
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.saveload import Checkpoint, load
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring

from CMUDict import CmuDict

from blocks_extras.extensions.plot import Plot


import theano
import numpy
from music_prepare import dataset
from theano import tensor

from bokeh.plotting import Session

class MusicRNNModel:

    def __init__(self, input_sources_list, input_sources_vocab_size_list,
                 output_source, output_source_vocab_size,
                 lookup_dim=200, hidden_size=256, recurrent_stack_size=1):

        self.InputSources = input_sources_list
        self.InputSourcesVocab = input_sources_vocab_size_list
        self.OutputSource = output_source
        self.OutputSourceVocab = output_source_vocab_size

        inputs = [tensor.lmatrix(source) for source in input_sources_list]
        output = tensor.lmatrix(output_source)

        lookups = self.get_lookups(lookup_dim, input_sources_vocab_size_list)

        for lookup in lookups:
            lookup.initialize()

        merge = Merge([lookup.name for lookup in lookups], [lookup.dim for lookup in lookups], hidden_size,
                              weights_init=initialization.Uniform(width=0.01),
                              biases_init=Constant(0))
        merge.initialize()

        linear0 = Linear(input_dim=hidden_size, output_dim=hidden_size,
                        weights_init=initialization.Uniform(width=0.01),
                        biases_init=Constant(0), name='linear0')
        linear0.initialize()

        recurrent_blocks = []

        for i in range(recurrent_stack_size):
            recurrent_blocks.append(SimpleRecurrent(
                dim=hidden_size, activation=Tanh(),
                weights_init=initialization.Uniform(width=0.01),
                use_bias=False))

        for i, recurrent_block in enumerate(recurrent_blocks):
            recurrent_block.name = 'recurrent'+str(i+1)
            recurrent_block.initialize()

        linear_out = Linear(input_dim=hidden_size, output_dim=output_source_vocab_size,
                              weights_init=initialization.Uniform(width=0.01),
                              biases_init=Constant(0), name='linear_out')
        linear_out.initialize()
        softmax = NDimensionalSoftmax(name='softmax')

        lookup_outputs = [lookup.apply(input) for lookup, input in zip(lookups, inputs)]

        m = merge.apply(*lookup_outputs)
        r = linear0.apply(m)
        for block in recurrent_blocks:
            r = block.apply(r)
        a = linear_out.apply(r)

        self.Cost = softmax.categorical_cross_entropy(output, a, extra_ndim=1).mean()
        self.Cost.name = 'cost'

        y_hat = softmax.apply(a, extra_ndim=1)
        y_hat.name = 'y_hat'

        self.ComputationGraph = ComputationGraph(self.Cost)

        self.Function = None
        self.MainLoop = None
        self.Model = Model(y_hat)

    def get_lookups(self, dim, vocab_list):
        return [LookupTable(dim=dim, length=vocab, name='lookup' + str(index),
                              weights_init=initialization.Uniform(width=0.01),
                              biases_init=Constant(0)) for index, vocab in enumerate(vocab_list)]

    def train(self, data_file, output_data_file, n_epochs=0):

        training_data = dataset.T_H5PYDataset(data_file, which_sets=('train',))
        test_data = dataset.T_H5PYDataset(data_file, which_sets=('test',))

        session = Session(root_url='http://localhost:5006')

        if self.MainLoop is None:
            step_rules = [RMSProp(learning_rate=0.2, decay_rate=0.95), StepClipping(1)]

            algorithm = GradientDescent(cost=self.Cost,
                                        parameters=self.ComputationGraph.parameters,
                                        step_rule=CompositeRule(step_rules),
                                        on_unused_sources='ignore')

            train_stream = DataStream.default_stream(
                training_data, iteration_scheme=SequentialScheme(
                    training_data.num_examples, batch_size=100))

            test_stream = DataStream.default_stream(
                test_data, iteration_scheme=SequentialScheme(
                    test_data.num_examples, batch_size=100))

            self.MainLoop = MainLoop(
                model=Model(self.Cost),
                data_stream=train_stream,
                algorithm=algorithm,
                extensions=[
                    FinishAfter(after_n_epochs=n_epochs),
                    Printing(),
                    Checkpoint(output_data_file, every_n_epochs=50),
                    TrainingDataMonitoring([self.Cost], after_batch=True, prefix='train'),
                    DataStreamMonitoring([self.Cost], after_batch=True, data_stream=test_stream, prefix='test'),
                    Plot(output_data_file, channels=[['train_cost', 'test_cost']])
                ])

        self.MainLoop.run()

    def get_var_from(self, name, vars):
        return vars[map(lambda x: x.name, vars).index(name)]

    def load(self, filename):
        self.MainLoop = load(open(filename))
        self.Model = self.MainLoop.model

        print self.Model.intermediary_variables

        model_inputs = [self.get_var_from(source, self.Model.variables) for source in self.InputSources]
        model_softmax = self.get_var_from('softmax_log_probabilities_output', self.Model.variables)

        parameter_dict = self.Model.get_parameter_dict()

        fun_updates = []

        for i in range(1,100):
            initial_state_key = "/recurrent"+str(i)+".initial_state"
            if initial_state_key not in parameter_dict:
                break
            intermediary_states = self.get_var_from("recurrent"+str(i)+"_apply_states", self.Model.intermediary_variables)
            fun_updates.append((parameter_dict[initial_state_key], intermediary_states[0][0]))

        self.Function = theano.function(model_inputs, model_softmax,
                                        updates=fun_updates)

    def sample(self, inputs_list):
        output = []
        out = 0

        for tup in zip(*inputs_list):
            new_tup = ()
            for p in tup:
                new_tup += ([[p]],)
            new_tup += ([[out]],)
            dist = numpy.exp(self.Function(*new_tup)[0])

            print dist

            out = numpy.random.choice(self.OutputSourceVocab, 1, p=dist)[0]
            output.append(out)

        return output


class MusicNetwork:

    def __init__(self, training_data_file):

        self.TrainingDataFile = training_data_file

        self.Dataset = dataset.T_H5PYDataset(training_data_file, ('train',))
        self.DurationsVocabSize = self.Dataset.durations_vocab_size()
        self.SyllablesVocabSize = self.Dataset.syllables_vocab_size()
        self.PitchesVocabSize = self.Dataset.pitches_vocab_size()
        self.PhraseVocabSize = self.Dataset.phrase_vocab_size()
        self.StressesVocabSize = 5

        self.PitchModel = MusicRNNModel(['durations', 'durations_prev', 'stress', 'phrase_end', 'pitches_prev'],
                                        [self.DurationsVocabSize, self.DurationsVocabSize, self.StressesVocabSize,
                                         self.PhraseVocabSize, self.PitchesVocabSize],
                                         'pitches', self.PitchesVocabSize)

        self.RhythmModel = MusicRNNModel(['stress', 'phrase_end', 'durations_prev'],
                                         [self.StressesVocabSize, self.PhraseVocabSize,
                                          self.DurationsVocabSize],
                                         'durations', self.DurationsVocabSize)

    def load(self):
        self.RhythmModel.load('trainingdata_rhythm.tar')
        self.PitchModel.load('trainingdata_pitches.tar')

    def sample(self, input_text):
        dict = CmuDict()
        phrase_end_list = []
        stress = []
        syllables = []
        for word in input_text.split(' '):
            end_phrase_word = len(word) > 0 and word[-1] in ['.', ',', '?', '!', ';', ')', ']']
            syllables.append(word)
            for idx, x in enumerate(list(dict.stress(word).replace('4', '_'))):
                stress += self.Dataset.stress_encode(x)
                if idx > 0:
                    syllables.append('')
                if end_phrase_word and idx == len(dict.stress(word)) - 1:
                    phrase_end_list.append(1)
                else:
                    phrase_end_list.append(0)

        print phrase_end_list

        rhythm_out = self.RhythmModel.sample([stress, phrase_end_list])
        pitches_out = self.PitchModel.sample([rhythm_out, [0]+rhythm_out[:-1], stress, phrase_end_list])
        return self.Dataset.durations_decode(rhythm_out), self.Dataset.pitches_decode(pitches_out), syllables