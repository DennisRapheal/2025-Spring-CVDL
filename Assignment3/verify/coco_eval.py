from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt = COCO("./gt.json")
dt = gt.loadRes("./../sample-image_result/pred.json")

evaluator = COCOeval(gt, dt, iouType='segm')
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()
