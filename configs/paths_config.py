dataset_paths = {
	'celeba_train': '',
	'celeba_test': '',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '',
	'etl_train': 'datasets/ETL_Test/train',
	'etl_test': 'datasets/ETL_Test/test',
	'etl_train_source': 'datasets/ETL_Test/train/source',
	'etl_train_target': 'datasets/ETL_Test/train/target',
	'etl_test_source': 'datasets/ETL_Test/test/source',
	'etl_test_target': 'datasets/ETL_Test/test/target',
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}
