__author__ = 'Steffen'

from blocks.bricks.recurrent import LSTM, RecurrentStack
from blocks.bricks import Tanh, Linear, NDimensionalSoftmax
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Merge
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.algorithms import GradientDescent, StepClipping, CompositeRule, Adam
from blocks.main_loop import MainLoop
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.saveload import Checkpoint

import theano.tensor


class ParallelCharModel:

    def __init__(self, input1_size, input2_size, lookup1_dim=200, lookup2_dim=200, hidden_size=512):
        self.hidden_size = hidden_size
        self.input1_size = input1_size
        self.input2_size = input2_size
        self.lookup1_dim = lookup1_dim
        self.lookup2_dim = lookup2_dim

        x1 = theano.tensor('char1')
        x2 = theano.tensor('char2')
        y = theano.tensor('outchar')

        lookup1 = LookupTable(dim=self.lookup1_dim, length=self.input1_size, name='lookup1')
        lookup2 = LookupTable(dim=self.lookup2_dim, length=self.input2_size, name='lookup2')
        l1 = lookup1.apply(x1)
        l2 = lookup2.apply(x2)

        merge = Merge(['lookup1', 'lookup2'], [self.lookup1_dim, self.lookup2_dim], self.hidden_size)
        m = merge.apply(l1, l2)

        recurrent_block = RecurrentStack([LSTM(dim=self.hidden_size, activation=Tanh())] * 3)
        h = recurrent_block.apply(m)

        linear = Linear(input_dim=self.hidden_size, output_dim=self.input1_size)
        a = linear.apply(h)

        softmax = NDimensionalSoftmax()

        y_hat = softmax.apply(a)

        self.Cost = softmax.categorical_cross_entropy(y, a, extra_ndim=1).mean()

        self.ComputationGraph = ComputationGraph(self.Cost)

        self.Model = Model(y_hat)

    def train(self, training_data):

        step_rules = [Adam(), StepClipping(1.0)]

        algorithm = GradientDescent(cost=self.Cost,
                                    parameters=self.ComputationGraph.parameters,
                                    step_rule=CompositeRule(step_rules),
                                    on_unused_sources='ignore')

        train_stream = DataStream.default_stream(
            training_data, iteration_scheme=SequentialScheme(
                training_data.num_examples, batch_size=20))

        main = MainLoop(
            model=Model(self.Cost),
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=[
                FinishAfter(),
                Printing(),
                Checkpoint('trainingdata.tar', every_n_epochs=10)
            ])

        main.run()

x = ParallelCharModel(100, 100)
x.train()