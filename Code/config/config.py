from easydict import EasyDict as edict

__C = edict()
cfg = __C


####### general parameters ######
__C.general = {}

####### config file in BME-cluster
__C.general_bme = {}
__C.general_bme.file_list = 'file_list.csv'
__C.general_bme.root = '/path/to/experiment/root'
__C.general_bme.save_root = '/path/to/save/training/Results'


###### training parameters
__C.train = {}
__C.train.num_epochs = 3000
__C.train.batch_size=1

__C.train.lr = 2e-4
__C.train.save_epoch = 1

# adaptive-weight for each brain region
__C.train.alpha = [0.4, 0.83480081, 0.835039219, 0.855442264, 0.856849387, 0.903668944, 0.899960582, 0.801572131, 0.805305151, 0.827592272, 0.826400611,
0.882260885, 0.88239142, 0.951479735, 0.948034505, 0.87013273, 0.870727194, 0.887590081, 0.886863638, 0.896744106, 0.902931944,
0.926612064, 0.917610825, 0.906178716, 0.899176097, 0.87300197, 0.871077487, 0.921040548, 0.929387065, 0.935497519, 0.92898779,
0.910577933, 0.908408125, 0.9172815, 0.919604719, 0.898028888, 0.895579268, 0.926219502, 0.927708938, 0.933756981, 0.930338289,
0.903949325, 0.90199541, 0.890742872, 0.888984014, 0.927821032, 0.928716227, 0.866409679, 0.868682666, 0.970833776, 0.965890817,
0.89972182, 0.90017701, 0.962348867, 0.960719271, 0.831378583, 0.839444537, 0.971178804, 0.974659603, 0.837054693, 0.838497367,
0.840393155, 0.830091603, 0.909936466, 0.907618845, 0.922725935, 0.926435268, 0.854779738, 0.854174813, 0.877002096, 0.874739101,
0.923646853, 0.919770513, 0.853524234, 0.85197966, 0.849148745, 0.855037396, 0.843107009, 0.845099628, 0.848506186, 0.842018638,
0.846684278, 0.84973009, 0.920368173, 0.919097022, 0.922910241, 0.924436804, 0.947432972, 0.954718038, 0.844152869, 0.845282211,
0.548154242, 0.548015698, 0.726321861, 0.725986707, 0.837524681, 0.841072484, 0.938602767, 0.930618603, 0.802044786, 0.502660583,
0.995234602, 0.952997488, 0.968289375, 0.96466205, 0.964637643, 0.948816911]

###### loss function setting ######
__C.loss = {}

__C.loss.name='HausdorffDistance'
__C.loss.weight = 0.4

# parameters for focal loss
__C.loss.obj_weight = ['99', '1']
__C.loss.gamma = 2

# resume_epoch == -1 training from scratch
__C.general.resume_epoch = -1

# random seed
__C.general.seed = 42


####### dataset parameters #######
__C.dataset = {}
# number of classes
__C.dataset.num_classes = 106
# number of modalities
__C.dataset.num_modalities = 1
# image resolution
__C.dataset.spacing = [1,1,1]
# cropped image patch size
__C.dataset.crop_size = [96, 96, 96]

# intensity normalize methods
# 1) FixedNormalize: use fixed mean and std to normalize image
# 2) AdaptiveNormalize: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.normalize = ['mean', 'std', 'is_clip']