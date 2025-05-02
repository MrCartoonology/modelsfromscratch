import modelsfromscratch.data as data
import modelsfromscratch.setup as setup
import modelsfromscratch.models as models
import modelsfromscratch.train as train


CFG_FNAME = setup.find_root() / "config/config.yaml"


def look_at_dataloader(
    dataloader: data.DataLoader, tokenizer: data.AutoTokenizer
) -> None:
    print(len(dataloader))
    x, y = next(iter(dataloader))
    print(x.shape)
    print(y.shape)
    print(x[0, 0:10])
    print(y[0, 0:10])
    print(tokenizer.decode(x[0, 0:10]))


def run(cfg_fname: str = CFG_FNAME):
    cfg = setup.load_config(cfg_fname)
    res = setup.RunTracker(cfg=cfg)
    res.tokenizer = data.get_tokenizer(cfg=cfg)
    res.dataloader = data.get_dataloader(res=res)
    res.model = models.load_model(res=res)
    train.train(res=res)


if __name__ == "__main__":
    run()
