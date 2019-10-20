from options.base_options import BaseOptions


class TransferOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.

    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--store_feature', type=str, default='/home/visiting/Projects/levishery/reconstruction/vangogh_features.json')
        parser.add_argument('--store_pca', type=str, default='/home/visiting/Projects/levishery/reconstruction/utils/vangogh.pca')
        parser.add_argument('--store_index', type=str, default='/home/visiting/Projects/levishery/reconstruction/vangogh_index.json')
        parser.add_argument('--result_dir', type=str, default='/home/visiting/Projects/levishery/reconstruction/result')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')


        self.isTrain = False
        return parser