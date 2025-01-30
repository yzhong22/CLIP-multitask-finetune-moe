from trainers.adaptation import AdaptationTrainer
from trainers.r_adapter import RAdapterTrainer

trainer_map = {
    "lp": AdaptationTrainer,
    "r-adapter": RAdapterTrainer,
    "cocoop": AdaptationTrainer,
    "taskres": AdaptationTrainer,
}


def build_trainer_func(args):
    return trainer_map[args.method]
