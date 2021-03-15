from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import pyspiel

from open_spiel.python.algorithms import cfr_br_actions
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import exploitability_br_actions
from open_spiel.python.algorithms import psro_oracle
from open_spiel.python.policy import tabular_policy_from_callable

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def create_random_tabular_policy(game, players=(0, 1)):
    def _random_action_callable_policy(state) -> dict:
        legal_actions_list = state.legal_actions()
        chosen_legal_action = np.random.choice(legal_actions_list)
        return {action: (1.0 if action == chosen_legal_action else 0.0) for action in legal_actions_list}

    return tabular_policy_from_callable(game=game, players=players, callable_policy=_random_action_callable_policy)


def policy_class_instance_to_full_policy(policy_class_instance):
    def wrap(state):
        action_probs = policy_class_instance.action_probabilities(state=state, player_id=state.current_player())
        ap_list = []
        for action, probs in action_probs.items():
            ap_list.append((action, probs))
        return ap_list

    return wrap


def get_random_population(number_of_strats, game):
    br_list = []
    for i in range(number_of_strats):
        brs = []
        for player in range(2):
            tabular_policy = create_random_tabular_policy(game, players=[player])
            full_policy = policy_class_instance_to_full_policy(tabular_policy)
            brs.append(full_policy)
        br_list.append(brs)
    return br_list


def get_xdo_restricted_game_meta_Nash(game, br_list, br_conv_threshold=1e-2, seed=1):
    episode = 0
    num_infostates = 0
    start_time = time.time()
    cfr_psro_times = []
    cfr_psro_exps = []
    cfr_psro_episodes = []
    cfr_psro_infostates = []

    cfr_br_solver = cfr_br_actions.CFRSolver(game, br_list)

    for j in range(int(1e10)):
        cfr_br_solver.evaluate_and_update_policy()
        episode += 1
        if j % 50 == 0:
            br_list_conv = exploitability_br_actions.exploitability(game, br_list, cfr_br_solver.average_policy())
            print("Br list conv: ", br_list_conv, j)
            conv = exploitability.exploitability(game, cfr_br_solver.average_policy())
            print("Iteration {} exploitability {}".format(j, conv))
            elapsed_time = time.time() - start_time
            print('Total elapsed time: ', elapsed_time)
            num_infostates = cfr_br_solver.num_infostates_expanded
            print('Num infostates expanded (mil): ', num_infostates / 1e6)
            cfr_psro_times.append(elapsed_time)
            cfr_psro_exps.append(conv)
            cfr_psro_episodes.append(episode)
            cfr_psro_infostates.append(num_infostates)

            save_prefix = './results/fixed_pop/XDO/num_pop_' + str(len(br_list)) + '_seed_' + str(seed)
            ensure_dir(save_prefix)
            print(f"saving to: {save_prefix + '_times.npy'}")
            np.save(save_prefix + '_times.npy', np.array(cfr_psro_times))
            print(f"saving to: {save_prefix + '_exps.npy'}")
            np.save(save_prefix + '_exps.npy', np.array(cfr_psro_exps))
            print(f"saving to: {save_prefix + '_episodes.npy'}")
            np.save(save_prefix + '_episodes.npy', np.array(cfr_psro_episodes))
            print(f"saving to: {save_prefix + '_infostates.npy'}")
            np.save(save_prefix + '_infostates.npy', np.array(cfr_psro_infostates))
            if br_list_conv < br_conv_threshold:
                print("Done")
                break


def get_psro_meta_Nash(game, br_list, num_episodes=100, seed=1):
    psro_br_list = []
    psro_br_list.append(br_list[0])
    psro_br_list.append([1 / len(br_list[0]) for _ in range(len(br_list[0]))])
    psro_br_list.append(br_list[1])
    psro_br_list.append([1 / len(br_list[1]) for _ in range(len(br_list[1]))])

    solver = psro_oracle.PSRO(game, psro_br_list, num_episodes=num_episodes)
    solver.evaluate()
    conv = exploitability.exploitability(game, solver._current_policy)
    save_path = './results/fixed_pop/PSRO/num_pop_'+ str(len(br_list)) + '_seed_' + str(seed) + '_exp.npy'
    print(f"saved to: {save_path}")
    ensure_dir(save_path)
    np.save(save_path, np.array(conv))
    print("PSRO Exploitability: ", conv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_strats', type=int, default=300, required=False)
    commandline_args = parser.parse_args()
    num_strats = commandline_args.num_strats
    game_name = 'leduc_poker'
    br_conv_threshold = 5e-2
    num_psro_episodes = 200
    seed = 1

    game = pyspiel.load_game(game_name, {"players": pyspiel.GameParameter(2)})
    br_list = get_random_population(num_strats, game)

    get_psro_meta_Nash(game, br_list, num_episodes=num_psro_episodes, seed=seed)
    get_xdo_restricted_game_meta_Nash(game, br_list, br_conv_threshold=br_conv_threshold, seed=seed)
