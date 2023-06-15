from types import SimpleNamespace

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))
cfg = SimpleNamespace(**{})
transform_config = {"stem": False, "lower": True, "stopwords": STOPWORDS}
cfg = SimpleNamespace(**transform_config)

resampling_config = {
    "strategy": {
        "train_test_split": {"train_size": 0.7, "random_state": 42, "shuffle": True}
    },
}
cfg.__dict__.update(resampling_config)
print(cfg.strategy["train_test_split"])
