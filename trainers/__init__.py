from trainers.adaptation import AdaptationTrainer
from trainers.r_adapter import RAdapterTrainer
from trainers.clap import CLAPTrainer
from trainers.flyp import FLYPTrainer

trainer_map = {
    "lp": AdaptationTrainer,
    "r-adapter": RAdapterTrainer,
    "cocoop": AdaptationTrainer,
    "taskres": AdaptationTrainer,
    "clap": CLAPTrainer,
    "wise-ft": AdaptationTrainer,
    "flyp": FLYPTrainer,
}


def build_trainer_func(args):
    return trainer_map[args.method]
