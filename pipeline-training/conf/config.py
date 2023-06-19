from rich.pretty import pprint

from conf.init_dirs import ROOT_DIR
from conf.init_project import initialize_project

cfg = initialize_project(ROOT_DIR)
pprint(cfg.env)
