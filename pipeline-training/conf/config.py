from conf.init_project import initialize_project
from conf.init_dirs import ROOT_DIR
from rich.pretty import pprint


cfg = initialize_project(ROOT_DIR)
pprint(cfg.env)
