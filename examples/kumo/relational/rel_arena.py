"""Kumo model submission using RelArena's shared validation tuner.

Use the same declared search space for every task. The candidate policy is
carried from stopped V12 development; complete runtime has not been certified.
"""

from examples.kumo.relational._relarena.adapter import KumoPredictor
from examples.kumo.relational._relarena.search_space import SEARCH_SPACE
from relarena.registry import register_model

KumoModel = register_model(search_space=SEARCH_SPACE)(KumoPredictor)
