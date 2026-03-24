# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.


def state_dict(self):
    """
    The state dict contains all non-DP-rank-dependent (i.e., non-parameter-
    related) optimizer variables. The returned state dict can be stored in
    the standard model/RNG checkpoint file. The parameter and dependent
    optimizer state (e.g., exp_avg, exp_avg_sq) are stored in a separate
    checkpoint file by calling 'save_parameter_state()'.
    """
    inner_state_dict = self.optimizer.state_dict()
    state_dict = {}

    # Extract 'step', for non-Apex/TE support.
    steps = list(set([s["step"].item() for s in inner_state_dict["state"].values()]))
    assert len(steps) == 1
    step = steps[0]

    # Optimizer state (do not store parameter state here).
    state_dict['optimizer'] = {k: v for k, v in inner_state_dict.items() if k != "state"}
    for param_group in state_dict["optimizer"]["param_groups"]:
        del param_group["params"]
        # Native PyTorch param group requires step (i.e., iteration).
        param_group["step"] = step

    if self.grad_scaler:
        state_dict['grad_scaler'] = self.grad_scaler.state_dict()

    return state_dict