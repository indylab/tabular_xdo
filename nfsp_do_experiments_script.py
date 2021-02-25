from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from random import shuffle
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import nashpy as nash
import random
import tensorflow as tf
import six

import open_spiel
from open_spiel.python.algorithms import cfr
# from open_spiel.python.algorithms import cfr_br_actions
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import exploitability_br_actions
from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.algorithms import fictitious_play_br_actions
# from open_spiel.python.algorithms import deep_cfr

from open_spiel.python import policy
from open_spiel.python.algorithms import nfsp


# from open_spiel.python.algorithms import lp_solver
# from open_spiel.python.algorithms import outcome_sampling_mccfr
# from open_spiel.python.algorithms import external_sampling_mccfr
# from open_spiel.python.algorithms import external_sampling_mccfr_br_actions
# from open_spiel.python.algorithms import outcome_sampling_mccfr_br_actions
# from open_spiel.python.algorithms import psro_oracle
from open_spiel.python import rl_environment
from open_spiel.python import rl_environment_br_actions


from open_spiel.python.algorithms import best_response as pyspiel_best_response
import pyspiel
import time
from numpy import array

game_name = "leduc_poker_dummy"
random_max_br = True
extra_info = 'seed1'
num_warm_starts = 20

game = pyspiel.load_game(game_name)
num_players = 2
env_configs = {"players": num_players}
pyspiel_game = pyspiel.load_game(game_name
                         ,{"players": pyspiel.GameParameter(2)})
env_configs = {"players": 2}
env = rl_environment.Environment(game, env_configs)



def _full_best_response_policy(br_infoset_dict):
    """Turns a dictionary of best response action selections into a full policy.

  Args:
    br_infoset_dict: A dictionary mapping information state to a best response
      action.

  Returns:
    A function `state` -> list of (action, prob)
  """

    def wrap(state):
        infostate_key = state.information_state_string(state.current_player())
        br_action = br_infoset_dict[infostate_key]
        ap_list = []
        for action in state.legal_actions():
            ap_list.append((action, 1.0 if action == br_action else 0.0))
        return ap_list

    return wrap

def _policy_dict_at_state(callable_policy, state):
    """Turns a policy function into a dictionary at a specific state.

  Args:
    callable_policy: A function from `state` -> lis of (action, prob),
    state: the specific state to extract the policy from.

  Returns:
    A dictionary of action -> prob at this state.
  """

    infostate_policy_list = callable_policy(state)
    infostate_policy = {}
    for ap in infostate_policy_list:
        infostate_policy[ap[0]] = ap[1]
    return infostate_policy

xfp_solver = fictitious_play.XFPSolver(game)
for i in range(1):
    xfp_solver.iteration()
    print('done with iter')
    conv = exploitability.exploitability(game, xfp_solver.average_policy())
    if i % 1 == 0:
        print("Iteration: {} Conv: {}".format(i, conv))
        

class NFSPPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies, mode):
        game = env.game
        player_ids = [0, 1]
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._mode = mode
        self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = (
            state.information_state_tensor(cur_player))
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)

        with self._policies[cur_player].temp_mode_as(self._mode):
            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


br_fulls = []
br_infos = []
info_test = []
for i in range(2):
    br_info = exploitability.best_response(pyspiel_game, xfp_solver.average_policy(), i)
#     print(br_info['best_response_action'], 'br info dict')
    full_br_policy = _full_best_response_policy(br_info["best_response_action"])
    info_sets = br_info['info_sets']
    info_test.append(info_sets)
    br_fulls.append(full_br_policy)
    br_infos.append(br_info)

br_full_list = [br_fulls]
br_info_list = [br_infos]


start_time = time.time()
nfsp_do_times = []
nfsp_do_episodes = []
nfsp_do_exps = []

br_conv_threshold = 2**num_warm_starts
iters = 100
episode = 0 
for i in range(iters):
    print('Iteration: ', i)
    
    env = rl_environment_br_actions.Environment(game, br_full_list, **env_configs)


    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in [128,]]
    kwargs = {
        "replay_buffer_capacity": int(2e5),
        "epsilon_decay_duration": int(3e6),
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
    }
    
    with tf.compat.v1.Session() as sess:
        # pylint: disable=g-complex-comprehension
        agents = [
            nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                      int(2e6), 0.1) for idx in range(num_players)
        ]
        expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

        sess.run(tf.compat.v1.global_variables_initializer())
        br_list_conv = 1000
        while br_list_conv > br_conv_threshold:
            if (episode + 1) % 5000 == 0 or i < num_warm_starts - 1:
                losses = [agent.loss for agent in agents]
#                 print("Losses: %s", losses)
#                 expl = exploitability.exploitability(env.game, expl_policies_avg)

#                 br_list_conv = exploitability_br_actions.exploitability(env.game, br_info_list, 
#                                                         expl_policies_avg)
                br_list_conv = exploitability_br_actions.exploitability(env.game, br_full_list, 
                                                    expl_policies_avg)


                print("Br list conv AVG: ", br_list_conv)
                print("_____________________________________________")

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)
    
            episode += 1
#             print('inner loop iteration: ', j)
            
        conv = exploitability.exploitability(env.game, expl_policies_avg)
    #             print(cfr_br_solver.average_policy())
        print("Iteration {} exploitability {}".format(i, conv))
        if conv < br_conv_threshold + .05:
            br_conv_threshold = br_conv_threshold*.5
            print('new br conv threshold: ', br_conv_threshold)

        elapsed_time = time.time() - start_time
        print('Total elapsed time: ', elapsed_time)
        print('Total episodes: ', episode)
        nfsp_do_times.append(elapsed_time)
        nfsp_do_episodes.append(episode)
        nfsp_do_exps.append(conv)
        np.save('./results/nfsp_do'+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_times', np.array(nfsp_do_times))
        np.save('./results/nfsp_do'+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_exps', np.array(nfsp_do_exps))
        np.save('./results/nfsp_do'+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_episodes', np.array(nfsp_do_episodes))

    
        br_fulls = []
        br_infos = []
        for i in range(2):
            if random_max_br:
                br_info = exploitability.best_response_random_max_br(pyspiel_game, expl_policies_avg, i)
            else:
                br_info = exploitability.best_response(pyspiel_game, expl_policies_avg, i)
            full_br_policy = _full_best_response_policy(br_info["best_response_action"])
            info_sets = br_info['info_sets']
            br_fulls.append(full_br_policy)
            br_infos.append(br_info)

        br_full_list.append(br_fulls)
        br_info_list.append(br_infos)
