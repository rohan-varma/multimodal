from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich import print

cli_conf = OmegaConf.from_cli()
yaml_conf = OmegaConf.load(cli_conf.config)
conf = instantiate(yaml_conf)
conf = OmegaConf.merge(conf, cli_conf)
print(conf.get("somethinginvalid", None))
