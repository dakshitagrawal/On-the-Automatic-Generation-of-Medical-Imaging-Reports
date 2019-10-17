from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
import json

def evalscores(hypotheses, references):
	targ_annotations = list()
	res_annotations = list()
	img_annotations = list()
	coco_ann_file = 'coco.json'
	res_ann_file = 'res.json'

	for i in range(len(hypotheses)):
		targ_anno_dict = {"image_id": i,
						  "id": i,
						  "caption": " ".join(references[i][0])}

		targ_annotations.append(targ_anno_dict)

		res_anno_dict = {"image_id": i,
						 "id": i,
						 "caption": " ".join(hypotheses[i])}

		res_annotations.append(res_anno_dict)

		image_anno_dict = {"id": i,
						   "file_name": i}

		img_annotations.append(image_anno_dict)

	coco_dict = {"type": 'captions', 
				 "images": img_annotations, 
				 "annotations": targ_annotations}

	res_dict = {"type": 'captions', 
				"images": img_annotations, 
				"annotations": res_annotations}

	with open(coco_ann_file, 'w') as fp:
		json.dump(coco_dict, fp)

	with open(res_ann_file, 'w') as fs:
		json.dump(res_annotations, fs)

	coco = COCO(coco_ann_file)
	cocoRes = coco.loadRes(res_ann_file)

	cocoEval = COCOEvalCap(coco, cocoRes)

	cocoEval.evaluate()

	for metric, score in cocoEval.eval.items():
		print('%s: %.3f'%(metric, score))