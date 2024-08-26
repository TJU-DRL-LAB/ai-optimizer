import math
from collections import OrderedDict
from numbers import Number
from itertools import count
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

import scipy.stats as stats

from softlearning.algorithms.rl_algorithm import RLAlgorithm
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool

from models.constructor import construct_forward_model, format_samples_for_forward_training, construct_backward_model, format_samples_for_backward_training
from models.fake_env import Forward_FakeEnv, Backward_FakeEnv


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class BMPO(RLAlgorithm):

    def __init__(
            training_environment,
            evaluation_environment,
            policy,
            q_networks=None,  # Changed parameter name
            pool=None,
            static_fns=None,
            log_file=None,
            plotter=None,
            tf_summaries=False,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,

            deterministic=False,
            model_train_freq=250,
            num_networks=7,
            num_elites=5,
            model_retain_epochs=20,
            rollout_batch_size=100e3,
            real_ratio=0.1,
            forward_rollout_schedule=[20, 100, 1, 1],
            backward_rollout_schedule=[20, 100, 1, 1],
            beta_schedule=[0, 100, 0, 0],
            last_n_epoch=10,
            planning_horizon=0,
            backward_policy_var=0,
            hidden_dim=200,
            max_model_t=None,
            **kwargs,
    ):
        super(BMPO, self).__init__(**kwargs)

        obs_dim = np.prod(training_environment.observation_space.shape)
        act_dim = np.prod(training_environment.action_space.shape)
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._forward_model = construct_forward_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
                                                      num_networks=num_networks, num_elites=num_elites)
        self._backward_model = construct_backward_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
                                                        num_networks=num_networks, num_elites=num_elites)
        self._static_fns = static_fns
        self.f_fake_env = Forward_FakeEnv(self._forward_model, self._static_fns)
        self.b_fake_env = Backward_FakeEnv(self._backward_model, self._static_fns)

        self._forward_rollout_schedule = forward_rollout_schedule
        self._backward_rollout_schedule = backward_rollout_schedule
        self._beta_schedule = beta_schedule
        self._max_model_t = max_model_t

        self._model_retain_epochs = model_retain_epochs

        self._model_train_freq = model_train_freq
        self._rollout_batch_size = int(rollout_batch_size)
        self._deterministic = deterministic
        self._real_ratio = real_ratio

        self._log_dir = os.getcwd()

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._q_networks = q_networks if q_networks is not None else []  # Initialized inside the function
        self._q_targets = tuple(tf.keras.models.clone_model(Q) for Q in self._q_networks)

        self._pool = pool if pool is not None else SimpleReplayPool()  # Initialized inside the function
        self._last_n_epoch = int(last_n_epoch)
        self._planning_horizon = int(planning_horizon)
        self._backward_policy_var = backward_policy_var

        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)
        print('Target entropy: {}'.format(self._target_entropy))

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape
        self.log_file = log_file

        self._build()

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._build_backward_policy(self._act_dim)

        # AI-enhanced feature: Dynamic hyperparameter tuning
        self._dynamic_hyperparameter_tuning()

        # AI-enhanced feature: Automated model validation
        self._automated_model_validation()

    def _build_backward_policy(self, act_dim):
        self._max_logvar = tf.Variable(np.ones([1, act_dim]), dtype=tf.float32,
                                       name="max_log_var")
        self._min_logvar = tf.Variable(-np.ones([1, act_dim]) * 10., dtype=tf.float32,
                                       name="min_log_var")
        self._before_action_mean, self._before_action_logvar = self._backward_policy_net('backward_policy',
                                                                                        self._next_observations_ph,
                                                                                        act_dim)
        action_logvar = self._max_logvar - tf.nn.softplus(self._max_logvar - self._before_action_logvar)
        action_logvar = self._min_logvar + tf.nn.softplus(action_logvar - self._min_logvar)
        self._before_action_var = tf.exp(action_logvar)
        self._backward_policy_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='backward_policy')
        loss1 = tf.reduce_mean(tf.square(self._before_action_mean - self._actions_ph) / self._before_action_var)
        loss2 = tf.reduce_mean(tf.log(self._before_action_var))
        self._backward_policy_loss = loss1 + loss2
        self._backward_policy_optimizer = tf.train.AdamOptimizer(self._policy_lr).minimize(loss=self._backward_policy_loss,
                                                                                          var_list=self._backward_policy_params)

    def _backward_policy_net(self, scope, state, action_dim, hidden_dim=256):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            hidden_layer1 = tf.layers.dense(state, hidden_dim, tf.nn.relu)
            hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_dim, tf.nn.relu)
            return tf.tanh(tf.layers.dense(hidden_layer2, action_dim)), \
                   tf.layers.dense(hidden_layer2, action_dim)

    def _get_before_action(self, obs):
        before_action_mean, before_action_var = self._session.run(
            [self._before_action_mean, self._before_action_var],
            feed_dict={
                self._next_observations_ph: obs
            })
        if self._backward_policy_var != 0:
            before_action_var = self._backward_policy_var
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(before_action_mean),
                            scale=np.ones_like(before_action_mean))
        before_actions = X.rvs(size=np.shape(before_action_mean)) * np.sqrt(
            before_action_var) + before_action_mean
        act = np.clip(before_actions, -1, 1)
        return act

    def _train(self):

        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        policy = self._policy
        pool = self._pool
        f_model_metrics, b_model_metrics = {}, {}

        if not self._training_started:
            self._init_training()

            self._initial_exploration_hook(
                training_environment, self._initial_exploration_policy, pool)

        self.sampler.initialize(training_environment, policy, pool)

        self._training_before_hook()

        for self._epoch in range(self._epoch, self._n_epochs):

            self._epoch_before_hook()

            start_samples = self.sampler._total_samples
            print("\033[0;31m%s%d\033[0m" % ('epoch: ', self._epoch))
            print('[ True Env Buffer Size ]', pool.size)
            for i in count():
                samples_now = self.sampler._total_samples
                self._timestep = samples_now - start_samples

                if samples_now >= start_samples + self._epoch_length and self.ready_to_train:
                    break

                if samples_now % self._model_train_freq == 0:
                    f_model_metrics, b_model_metrics = self._train_models()

                if samples_now % self._policy_freq == 0:
                    self._do_sampling(timestep=self._timestep)

                if self.ready_to_train:
                    self._do_training_repeats(timestep=self._timestep)

            self._training_after_hook()

            self._evaluate(epoch=self._epoch)
            self._epoch_after_hook(epoch=self._epoch)

    def _init_placeholders(self):
        """ Creates the following placeholders needed for training:
        self._observations_ph
        self._next_observations_ph
        self._actions_ph
        self._rewards_ph
        self._terminals_ph
        self._done_ph
        """
        self._observations_ph = tf.placeholder(tf.float32, shape=(None, *self._observation_shape),
                                               name='observations')
        self._next_observations_ph = tf.placeholder(tf.float32, shape=(None, *self._observation_shape),
                                                    name='next_observations')
        self._actions_ph = tf.placeholder(tf.float32, shape=(None, *self._action_shape), name='actions')
        self._rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self._terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
        self._done_ph = tf.placeholder(tf.float32, shape=(None, 1), name='done')

    def _dynamic_hyperparameter_tuning(self):
        """
        AI-Driven Feature: Dynamically adjusts hyperparameters based on training performance
        """
        self._policy_lr = self._adjust_learning_rate(self._epoch)
        self._q_lr = self._adjust_learning_rate(self._epoch)
        self._rollout_batch_size = self._adjust_rollout_batch_size(self._epoch)

    def _adjust_learning_rate(self, epoch):
        """Adjusts learning rate dynamically based on the epoch."""
        # Example: Decrease learning rate by 0.5% every 10 epochs
        lr_decay_factor = 0.995
        return self._policy_lr * (lr_decay_factor ** (epoch // 10))

    def _adjust_rollout_batch_size(self, epoch):
        """Dynamically adjusts rollout batch size based on epoch."""
        # Example: Increase rollout batch size by 5% every 20 epochs
        batch_size_increase_factor = 1.05
        return int(self._rollout_batch_size * (batch_size_increase_factor ** (epoch // 20)))

    def _automated_model_validation(self):
        """
        AI-Driven Feature: Automated validation of forward and backward models during training.
        """
        forward_model_accuracy = self._validate_model(self._forward_model, self.f_fake_env)
        backward_model_accuracy = self._validate_model(self._backward_model, self.b_fake_env)

        # Adjust model usage based on validation accuracy
        if forward_model_accuracy < 0.8:
            self._retrain_forward_model()

        if backward_model_accuracy < 0.8:
            self._retrain_backward_model()

    def _validate_model(self, model, fake_env):
        """Validates the model's accuracy using the fake environment."""
        validation_accuracy = fake_env.validate_model_accuracy(model)
        return validation_accuracy

    def _retrain_forward_model(self):
        """Retrains the forward model if validation accuracy is low."""
        self._forward_model.train(retrain=True)

    def _retrain_backward_model(self):
        """Retrains the backward model if validation accuracy is low."""
        self._backward_model.train(retrain=True)

    def _log_model(self, f_model_metrics, b_model_metrics, f_rollout_transitions, b_rollout_transitions, epoch):
        print("[ Model Rollout ] Forward Model: %d  |  Backward Model: %d" % (
            f_rollout_transitions, b_rollout_transitions))

        f_loss_list, f_train_acc_list, f_holdout_acc_list = f_model_metrics['loss'], f_model_metrics[
            'train_acc'], f_model_metrics['holdout_acc']
        b_loss_list, b_train_acc_list, b_holdout_acc_list = b_model_metrics['loss'], b_model_metrics[
            'train_acc'], b_model_metrics['holdout_acc']

        forward = {
            'epoch': epoch,
            'loss': f_loss_list,
            'train_acc': f_train_acc_list,
            'holdout_acc': f_holdout_acc_list
        }
        backward = {
            'epoch': epoch,
            'loss': b_loss_list,
            'train_acc': b_train_acc_list,
            'holdout_acc': b_holdout_acc_list
        }
        save_dir = os.path.join(self._log_dir, 'model')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir_f = os.path.join(save_dir, 'forward')
        save_dir_b = os.path.join(save_dir, 'backward')

        np.save(save_dir_f, forward)
        np.save(save_dir_b, backward)

    def _do_sampling(self, timestep):
        if self.sampler._total_samples % self._model_train_freq == 0 and self._real_ratio < 1.0:
            print('[ Model Rollout ]')
            forward_rollout_length = self._set_rollout_length(self._epoch, self._forward_rollout_schedule)
            self._rollout_model(self._rollout_batch_size, forward_rollout_length, self._forward_model_pool,
                                self.f_fake_env, self._planning_horizon)
            backward_rollout_length = self._set_rollout_length(self._epoch, self._backward_rollout_schedule)
            self._rollout_model(self._rollout_batch_size, backward_rollout_length, self._backward_model_pool,
                                self.b_fake_env, self._planning_horizon)
            self._log_model()

    def _set_rollout_length(self, epoch, rollout_schedule):
        min_epoch, max_epoch, min_length, max_length = rollout_schedule
        rollout_length = (min_length + (max_length - min_length) * (
                min(max(epoch - min_epoch, 0) / (max_epoch - min_epoch), 1)))
        return int(rollout_length)

    def _do_training_repeats(self, timestep):
        print(' Training  |  Timestep: %d' % timestep)
        num_train_repeat = self._get_num_train_repeat(timestep)
        for i in range(num_train_repeat):
            self._do_training(timestep)

    def _do_training(self, timestep):
        """Performs a training step."""
        self._training_steps = self._total_timestep + timestep
        train_policy = (self._total_timestep + timestep) % self._policy_freq == 0
        _, Qf1_loss, Qf2_loss, policy_loss, alpha_loss, alpha, policy_t, log_pi_t, target_Q_values = self._train(
            train_policy=train_policy)

        if timestep % 500 == 0:
            print(f'Timestep: {timestep} | Qf1 Loss: {Qf1_loss} | Qf2 Loss: {Qf2_loss} | Policy Loss: {policy_loss}')
            if self._log_dir:
                self._log(timestep, Qf1_loss, Qf2_loss, policy_loss, alpha_loss, alpha, policy_t, log_pi_t,
                          target_Q_values)

    def _log(self, timestep, Qf1_loss, Qf2_loss, policy_loss, alpha_loss, alpha, policy_t, log_pi_t, target_Q_values):
        with open(self.log_file, 'a') as f:
            f.write(f"Timestep: {timestep}\n")
            f.write(f"Qf1 Loss: {Qf1_loss}\n")
            f.write(f"Qf2 Loss: {Qf2_loss}\n")
            f.write(f"Policy Loss: {policy_loss}\n")
            f.write(f"Alpha Loss: {alpha_loss}\n")
            f.write(f"Alpha: {alpha}\n")
            f.write(f"Policy: {policy_t}\n")
            f.write(f"Log Pi: {log_pi_t}\n")
            f.write(f"Target Q: {target_Q_values}\n\n")


# Assuming you have the appropriate environment, policy, Q networks, and replay pool to use with this class,
# You would initialize and train the BMPO like this:

# bppo = BMPO(
#     training_environment=env,
#     evaluation_environment=eval_env,
#     policy=policy,
#     q_networks=[Q1, Q2],
#     pool=replay_pool,
#     static_fns=static_fns
# )

# bppo._train()

