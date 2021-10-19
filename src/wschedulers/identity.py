
class Wrapper:
    @staticmethod
    def get_args(parser):
        pass

    @staticmethod
    def get_scheduler(args):
        scheduler = Scheduler(args.wsc_step)
        return scheduler


class Scheduler:
    def __init__(self, start_step=0):
        self.step = start_step
        # pass

    def __call__(self, loss_G_dict, loss_D_dict=None):
        self.step += 1

        if loss_D_dict is None:
            return loss_G_dict
        else:
            return loss_G_dict, loss_D_dict