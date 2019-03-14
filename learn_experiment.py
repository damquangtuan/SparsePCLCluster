from cluster_work import ClusterWork
from trainer import Trainer

class MyExperiment(ClusterWork):
    def __init__(self):
        ClusterWork.__init__(self)
        self.trainer_obj = Trainer()
        self.file_to_save = None

    def reset(self, config, rep=0):
        """
        Run code that sets up repetition rep of  your experiment.
        Won't be implemented for each iteration!

        :param config: a dictionary with the experiment configuration
        :param rep: the repetition counter
        """

        self.file_to_save = '/home/qd34dado/Workspace/SparsePCL/results/Copy-v0_20_q_' + str(config['param']['q'])\
                        + '_tau_' + str(config['param']['tau']) + '_learning_rate_0.01'

        self.train_obj.dqn = Trainer(
            batch_size=config['params']['batch_size'],
            validation_frequency=25,
            rollout=10,
            critic_weight=1.0,
            gamma=0.9,
            clip_norm=10,
            replay_buffer_freq=1,
            objective='generaltsallisv2',
            learning_rate=config['params']['learning_rate'],
            q=config['params']['q'],
            k=config['params']['k'],
            tsallis=True,
            tau=config['params']['tau']
        )


    def iterate(self, config, rep=0, n=0):
        """
        Run iteration n of repetition rep of your experiment.

        :param config: a dictionary with the experiment configuration
        :param rep: the repetition counter
        :param n: the iteration counter
        """
        file_to_save = self.file_to_save + str(n) + '.txt'
        self.trainer_obj.set_file_to_save(file_to_save=file_to_save)

        self.trainer_obj.run()
        # Return results as a dictionary, for each key there will be one column in a results pandas.DataFrame.
        # The DataFrame will be stored below the path defined in the experiment config.
        return {'results': None}

# to run the experiments, you simply call run on your derived class
if __name__ == '__main__':
    MyExperiment.run()
