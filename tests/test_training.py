from project_x.training.train import repeat_dataloader


class EpochLoader:
    def __init__(self):
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        yield self.epoch, "first"
        yield self.epoch, "second"


def test_repeat_dataloader_restarts_without_caching_an_epoch():
    loader = EpochLoader()
    iterator = repeat_dataloader(loader)

    assert [next(iterator) for _ in range(5)] == [
        (1, "first"),
        (1, "second"),
        (2, "first"),
        (2, "second"),
        (3, "first"),
    ]
