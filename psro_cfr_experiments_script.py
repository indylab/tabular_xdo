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
import argparse

import open_spiel
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import cfr_br_actions
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import exploitability_br_actions
from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.algorithms import fictitious_play_br_actions
from open_spiel.python import policy

from open_spiel.python.algorithms import lp_solver
from open_spiel.python.algorithms import outcome_sampling_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr_br_actions
from open_spiel.python.algorithms import psro_oracle

from open_spiel.python.algorithms import best_response as pyspiel_best_response
import pyspiel
import time
import datetime
from numpy import array

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, choices=["psro", "mccfr", "cfr", "xfp", "xdo"])
parser.add_argument('--game_name', type=str, required=False, default="leduc_poker", choices=["leduc_poker", "kuhn_poker", "leduc_poker_dummy"])
commandline_args = parser.parse_args()

algorithm = commandline_args.algorithm
game_name = commandline_args.game_name
extra_info = datetime.datetime.now().strftime("%I.%M.%S%p_%b-%d-%Y")
game = pyspiel.load_game(game_name, 
                        {"players": pyspiel.GameParameter(2)})
starting_br_conv_threshold = 2**4
iterations = 1000000
xdo_iterations = 200000
random_max_br = False


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

def run(solver, iterations):
    start_time = time.time()
    times = []
    exps = []
    episodes = []
    cfr_infostates = []
    for i in range(iterations):
        if algorithm == 'cfr':
            solver.evaluate_and_update_policy()
        else:
            solver.iteration()
        if i % 5 == 0:
            print(algorithm)
            if algorithm == 'mccfr':
                average_policy = policy.PolicyFromCallable(game, solver.callable_avg_policy())
            elif algorithm == 'cfr':
                average_policy = solver.average_policy()
            elif algorithm == 'xfp':
                average_policy = solver.average_policy()
            elif algorithm == 'psro':
                average_policy = solver._current_policy
            conv = exploitability.exploitability(game, average_policy)
            print("Iteration {} exploitability {}".format(i, conv))
            elapsed_time = time.time() - start_time
            print(elapsed_time)
            times.append(elapsed_time)
            exps.append(conv)
            episodes.append(i)
            np.save('./results/'+algorithm+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_times', np.array(times))
            np.save('./results/'+algorithm+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_exps', np.array(exps))
            np.save('./results/'+algorithm+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_episodes', np.array(episodes))
            if algorithm == 'cfr':
                cfr_infostates.append(solver.num_infostates_expanded)
                print("Num infostates expanded (mil): ", solver.num_infostates_expanded/1e6)
                np.save('./results/'+algorithm+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_infostates', np.array(cfr_infostates))


if algorithm == 'cfr':
    solver = cfr.CFRSolver(game)
    run(solver, iterations)
elif algorithm == 'mccfr':
    solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
    run(solver, iterations)
elif algorithm == 'xfp':
    solver = fictitious_play.XFPSolver(game)
    run(solver, iterations)
elif algorithm == 'xdo':
    brs = []
    info_test = []
    for i in range(2):
        br_info = exploitability.best_response(game, cfr.CFRSolver(game).average_policy(), i)
        full_br_policy = _full_best_response_policy(br_info["best_response_action"])
        info_sets = br_info['info_sets']
        info_test.append(info_sets)
        brs.append(full_br_policy)
    br_list = [brs]
    start_time = time.time()
    xdo_times = []
    xdo_exps = []
    xdo_episodes = []
    xdo_infostates = []

    br_conv_threshold = starting_br_conv_threshold

    episode = 0
    num_infostates = 0
    for i in range(iterations):
        print('Iteration: ', i)
        cfr_br_solver = cfr_br_actions.CFRSolver(game, br_list)

        for j in range(xdo_iterations):
            cfr_br_solver.evaluate_and_update_policy()
            episode += 1
            if j%50==0:
                br_list_conv = exploitability_br_actions.exploitability(game, br_list, cfr_br_solver.average_policy())
                print("Br list conv: ", br_list_conv, j)
                if br_list_conv < br_conv_threshold:
                    break
            

        conv = exploitability.exploitability(game, cfr_br_solver.average_policy())
        print("Iteration {} exploitability {}".format(i, conv))
        if conv < br_conv_threshold:
            br_conv_threshold /= 2
            print("new br threshold: ", br_conv_threshold)

        elapsed_time = time.time() - start_time
        print('Total elapsed time: ', elapsed_time)
        num_infostates += cfr_br_solver.num_infostates_expanded
        print('Num infostates expanded (mil): ', num_infostates/1e6)
        xdo_times.append(elapsed_time)
        xdo_exps.append(conv)
        xdo_episodes.append(episode)
        xdo_infostates.append(num_infostates)

        brs = []
        for i in range(2):
            if random_max_br:
                br_info = exploitability.best_response_random_max_br(game, cfr_br_solver.average_policy(), i)
            else:
                br_info = exploitability.best_response(game, cfr_br_solver.average_policy(), i)
            full_br_policy = _full_best_response_policy(br_info["best_response_action"])
            brs.append(full_br_policy)

        br_list.append(brs)
        np.save('./results/'+algorithm+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_times', np.array(xdo_times))
        np.save('./results/'+algorithm+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_exps', np.array(xdo_exps))
        np.save('./results/'+algorithm+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_episodes', np.array(xdo_episodes))
        np.save('./results/'+algorithm+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_infostates', np.array(xdo_infostates))

        
elif algorithm == 'xfp_psro':
    brs = []
    info_test = []
    for i in range(2):
        br_info = exploitability.best_response(game, cfr.CFRSolver(game).average_policy(), i)
        full_br_policy = _full_best_response_policy(br_info["best_response_action"])
        info_sets = br_info['info_sets']
        info_test.append(info_sets)
        brs.append(full_br_policy)
    br_list = [brs]
    start_time = time.time()
    xfp_psro_times = []
    xfp_psro_exps = []
    xfp_psro_episodes = []

    br_conv_threshold = starting_br_conv_threshold

    episode = 0
    for i in range(iterations):
        print('Iteration: ', i)
        xfp_br_solver = fictitious_play_br_actions.XFPSolver(game, br_list)

        for j in range(xdo_iterations):
            xfp_br_solver.iteration()
            episode += 1
            if j%50==0:
                br_list_conv = exploitability_br_actions.exploitability(game, br_list, xfp_br_solver.average_policy())
                print("Br list conv: ", br_list_conv, j)
                if br_list_conv < br_conv_threshold:
                    break

        conv = exploitability.exploitability(game, xfp_br_solver.average_policy())
        print("Iteration {} exploitability {}".format(i, conv))
        if conv < br_conv_threshold:
            br_conv_threshold /= 2
            print("new br threshold: ", br_conv_threshold)

        elapsed_time = time.time() - start_time
        print('Total elapsed time: ', elapsed_time)
        xfp_psro_times.append(elapsed_time)
        xfp_psro_exps.append(conv)
        xfp_psro_episodes.append(episode)

        brs = []
        for i in range(2):
            if random_max_br:
                br_info = exploitability.best_response_random_max_br(game, xfp_br_solver.average_policy(), i)
            else:
                br_info = exploitability.best_response(game, xfp_br_solver.average_policy(), i)
            full_br_policy = _full_best_response_policy(br_info["best_response_action"])
            brs.append(full_br_policy)

        br_list.append(brs)
        np.save('./results/'+algorithm+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_times', np.array(xfp_psro_times))
        np.save('./results/'+algorithm+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_exps', np.array(xfp_psro_exps))
        np.save('./results/'+algorithm+'_'+game_name+'_random_br_'+str(random_max_br)+extra_info+'_episodes', np.array(xfp_psro_episodes))

        
elif algorithm == 'psro':
    brs = []
    info_test = []
    for i in range(2):
        br_info = exploitability.best_response(game, cfr.CFRSolver(game).average_policy(), i)
        full_br_policy = _full_best_response_policy(br_info["best_response_action"])
        info_sets = br_info['info_sets']
        info_test.append(info_sets)
        brs.append(full_br_policy)
    br_list = [[brs[0]], [1], [brs[1]], [1]]  
    solver = psro_oracle.PSRO(game, br_list, num_episodes=2000)
    run(solver, iterations)
    



