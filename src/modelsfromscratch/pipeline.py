"""Pipeline for running the model training and evaluation process."""

from modelsfromscratch.data import get_dataloaders, get_tokenizer
from modelsfromscratch.models import load_model
from modelsfromscratch.setup import find_root, load_config, RunTracker
from modelsfromscratch.train import train

import logging

CFG_FNAME = find_root() / "config/config.yaml"


def run(cfg_fname: str = CFG_FNAME) -> RunTracker:
    """
    Run the complete training pipeline.

    Args:
        cfg_fname: Path to the configuration file. Defaults to the default config path.

    Returns:
        RunTracker: The run tracker object containing the trained model and results.
    """
    try:
        cfg = load_config(cfg_fname)
        res = RunTracker(cfg=cfg)
        
        # Setup tokenizer and dataloaders
        res.tokenizer = get_tokenizer(cfg=cfg)
        res.dataloaders = get_dataloaders(res=res)
        
        # Load and train model
        res.model = load_model(res=res)
        res = train(res=res)
        
        return res
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    run()
