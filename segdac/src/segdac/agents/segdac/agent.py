from segdac.agents.sac.agent import SacAgent


class SegdacSacAgent(SacAgent):
    """
    SegDAC uses standard SAC as base algo but other algos can be used (might require different hyperparameters).
    To see which network classes are used for the actor and critic you can check configs/algo/segdac.yaml
    """
    pass

