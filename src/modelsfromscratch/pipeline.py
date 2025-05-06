import modelsfromscratch.data as data
import modelsfromscratch.setup as setup
import modelsfromscratch.models as models
import modelsfromscratch.train as train


CFG_FNAME = setup.find_root() / "config/config.yaml"


def run(cfg_fname: str = CFG_FNAME):
    cfg = setup.load_config(cfg_fname)
    res = setup.RunTracker(cfg=cfg)
    res.tokenizer = data.get_tokenizer(cfg=cfg)
    res.dataloaders = data.get_dataloaders(res=res)
    res.model = models.load_model(res=res)
    train.train(res=res)


if __name__ == "__main__":
    run()
