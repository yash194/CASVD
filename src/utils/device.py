import torch as th


def is_mps_available():
    return hasattr(th.backends, "mps") and th.backends.mps.is_available()


def sanitise_device_config(config, log):
    config.setdefault("use_cuda", False)
    config.setdefault("use_mps", False)

    if config["use_cuda"] and config["use_mps"]:
        log.warning("Both use_cuda and use_mps were set. CUDA will be preferred when available.")

    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["use_mps"] and not is_mps_available():
        config["use_mps"] = False
        log.warning("MPS flag use_mps was switched OFF automatically because MPS is not available!")

    return config


def get_device_name(config):
    if config.get("use_cuda", False):
        return "cuda"
    if config.get("use_mps", False):
        return "mps"
    return "cpu"


def move_optimizer_state(optimizer, device):
    if optimizer is None:
        return

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, th.Tensor):
                state[key] = value.to(device)
