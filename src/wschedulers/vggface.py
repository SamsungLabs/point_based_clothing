
class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--start_step', type=int, default=0)
        parser.add('--step_size', type=float, default=(2e-2)/70)
        parser.add('--step_freq', type=int, default=1)
        parser.add('--maxmult', type=float, default=1.)

    @staticmethod
    def get_scheduler(args):
        if not hasattr(args, 'wsc_step'):
            args.wsc_step = int((args.maxmult / args.step_size) * args.step_freq)

        scheduler = Scheduler(args.start_step, args.step_size, args.step_freq, args.maxmult, args.wsc_step)
        return scheduler


class Scheduler:
    def __init__(self, start_step, step_size, step_freq, maxmult, current_step=0):
        self.step = current_step
        self.start_step = start_step
        self.step_size = step_size
        self.step_freq = step_freq
        self.maxmult = maxmult
        # pass

    def __call__(self, loss_G_dict, loss_D_dict=None):
        n_step = (self.step - self.start_step) / self.step_freq
        n_step = max(n_step, 0)

        multiplier = self.step_size * n_step
        multiplier = min(multiplier, self.maxmult)

        loss_G_dict_weighted = dict()
        loss_G_dict_weighted.update(loss_G_dict)
        loss_G_dict_weighted['vgg_face'] = loss_G_dict_weighted['vgg_face']*multiplier

        self.step += 1

        if loss_D_dict is None:
            return loss_G_dict_weighted
        else:
            return loss_G_dict_weighted, loss_D_dict

        