from tensorboardX import SummaryWriter

class LogSummary():

    def __init__(self,name):
        self.writer = SummaryWriter('logs/' + name, flush_secs=1)

    def write_final_accuracy(self, accuracy, fold, epoch):
        self.writer.add_scalar('fold{}-TestAccuracy'.format(fold), accuracy, epoch)

    def write_per_round_accuracy(self, accuracy, round, epoch):
        self.writer.add_scalar('Round/'+str(round)+'/Accuracy', accuracy, epoch)
    
    def per_round_layer_output(self, layer_sz, layer_op, round):
        self.writer.add_histogram('/Layer-'+str(layer_sz), layer_op, round)
