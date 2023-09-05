from argparse import Namespace
import yaml

def initialize_arguments(parser):

     parser.add_argument('-config', help='configuration file *.yml', \
          type=str, required=True, default='None')
     parser.add_argument('-m', '--model', help='model name 1.mlp, 2.tabmlp', \
          default='mlp')
     parser.add_argument('-test', '--test', help='if you want to run in test mode', \
          action='store_true')
     parser.add_argument('-fbs', '--file_b_sz', help='batch size for reading files', \
                          default=1, type=int)
     parser.add_argument('-bs', '--b_sz', help='batch size', default=256, type=int)
     parser.add_argument('-train_path', help='the complete path of data', required=False)
     parser.add_argument('-valid_path', help='the complete path of data', required=False)
     parser.add_argument('-test_path', help='the complete path of data', required=False)
     parser.add_argument('-extension', help='the type of file extension csv or parquet', required=False)
     parser.add_argument('-model_storage_path', help='the complete path of data', required=False)
     parser.add_argument('-e', '--epochs', help='number of epochs', default=150, type=int)
     parser.add_argument('-lr', '--learning_rate', help='learning rate', default=0.001, type=float)
     parser.add_argument('-op', '--optimizer', help='optimizer types, 1. SGD 2. Adam, \
          default SGD', default='Adam')
     parser.add_argument('-ba', '--is_bayesian', help='to use bayesian \
          layer or not', action='store_true')
     parser.add_argument('-is_valid', help='user validation data or not: 1 or 0 respectively', \
          type=int, default=0)
     parser.add_argument('-inference_model', help='set to True if training, else False', \
         type=bool, default=False)
     parser.add_argument('-r', '--resume', help='if you want to resume from an epoch', \
          action='store_true')
     parser.add_argument('-patience', help='for early stopping. How many epochs to wait', \
          default=10, type=int)
     parser.add_argument('-report_test', help='if you want test the model at every training \
          epoch (disabling this will reduce moel training time)', action='store_true')
     parser.add_argument('-ckpt_path', help='Path to the checkpoint file, if you want \
          to load the pre-trained state of the model', required=False, default='None', type=str)
     parser.add_argument('-num_feat_path', help='required numerical features', required=False, type=int)
     parser.add_argument('-file_workers', help='# workers for reading files', required=False, 
                         default=1, type=int)
     parser.add_argument('-data_workers', help='# workers to for loading data', required=False, 
                         default=4, type=int)
     parser.add_argument('-categ_feat_path', help='required categorical features', required=False, type=str)
     parser.add_argument('-target_label', help='name of the target feature column', \
          required=False, type=str)
     args = parser.parse_args()

     if args.config != 'None':
          opt = vars(args)
          yaml_content = yaml.load(open(args.config), Loader=yaml.FullLoader)

          # Updated reading from the nested YAML structure
          opt.update(yaml_content['model_parameters'])
          opt.update(yaml_content['data_parameters'])
          opt.update(yaml_content['execution_parameters'])
          opt.update(yaml_content['performance_parameters'])
          args = Namespace(**opt)
     return args

     # if args.config != 'None':
     #      opt = vars(args)
     #      args = yaml.load(open(args.config), Loader=yaml.FullLoader)
     #      opt.update(args)
     #      args = Namespace(**opt)
     # return args