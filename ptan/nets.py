import copy


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights 
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())
