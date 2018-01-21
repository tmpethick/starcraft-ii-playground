# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import csv

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

_ACTION_LENGTH = 3

class CollectMineralShards(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""

  def step(self, obs):
    super(CollectMineralShards, self).step(obs)
    action_id = None
    position = None
    
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      if not neutral_y.any() or not player_y.any():
        return actions.FunctionCall(_NO_OP, [])
      player = [int(player_x.mean()), int(player_y.mean())]
      closest, min_dist = None, None
      for p in zip(neutral_x, neutral_y):
        dist = np.linalg.norm(np.array(player) - np.array(p))
        if not min_dist or dist < min_dist:
          closest, min_dist = p, dist
      action_id = 0
      position = closest
      action = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
    else:
      action_id = 1
      action = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
    
    screen = obs.observation["screen"][_PLAYER_RELATIVE]

    discretized_pos = np.zeros(screen.shape)
    if position:
      discretized_pos[position[0], position[1]] = 1

    with open('X.csv','ab') as out:
      save_row(out, screen)

    with open('Y-spacial.csv','ab') as out:
      save_row(out, discretized_pos)

    with open('Y.csv','ab') as out:
      save_row(out, np.array([action_id]))

    return action

def save_row(file_handler, matrix):
  np.savetxt(file_handler, matrix.reshape(1, matrix.size), delimiter=',', fmt="%.d")
