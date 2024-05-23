import os
import yaml
import sys
import subprocess

import hydra
from omegaconf import OmegaConf, DictConfig

from jinja2 import Environment, FileSystemLoader

from dataclasses import dataclass

@hydra.main(config_path="hydra_config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # ----------------------------------------------------------------------- #
    #  Share values
    # ----------------------------------------------------------------------- #
    job = cfg.job

    # -- Job
    cfg.path.file_yaml_job = f"{job}.yaml"
    cfg.path.file_bsub_job = f"{job}.bsub"
    cfg.bsub_config.job    = cfg.job

    # -- Yaml
    os.makedirs(cfg.path.dir_yaml_jobs, exist_ok = True)
    path_yaml = os.path.join(cfg.path.dir_yaml_jobs, cfg.path.file_yaml_job)
    cfg.bsub_config.yaml_config = path_yaml

    # ----------------------------------------------------------------------- #
    #  YAML
    # ----------------------------------------------------------------------- #
    print("Generated YAML Script:")
    print(OmegaConf.to_yaml(cfg.predict_config))
    print()

    path_yaml_job = os.path.join(cfg.path.dir_yaml_jobs, cfg.path.file_yaml_job)
    with open(path_yaml_job, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg.predict_config))

        fh.write("\n")
        end_note = '# THIS SCRIPT IS GENERATED BY EXECUTING: \n# python ' + ' '.join(sys.argv)
        fh.write(end_note)


    # ----------------------------------------------------------------------- #
    #  BSUB
    # ----------------------------------------------------------------------- #
    env             = Environment(loader=FileSystemLoader(cfg.path.dir_bsub_template))
    template        = env.get_template(cfg.path.file_bsub_template)
    rendered_script = template.render(**cfg.bsub_config)

    print("Generated BSUB Script:")
    print(rendered_script)
    print()

    os.makedirs(cfg.path.dir_bsub_jobs, exist_ok = True)
    path_bsub_job = os.path.join(cfg.path.dir_bsub_jobs, cfg.path.file_bsub_job)
    with open(path_bsub_job, 'w') as fh:
        fh.write(rendered_script)

        fh.write("\n")
        end_note = '# THIS SCRIPT IS GENERATED BY EXECUTING: \n# python ' + ' '.join(sys.argv)
        fh.write(end_note)

    if cfg.auto_submit:
        # Executing the bsub script
        bsub_command = f"bsub {path_bsub_job}"
        print(bsub_command)
        os.system(bsub_command)

if __name__ == "__main__":
    main()
