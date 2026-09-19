from mot_pa2.evaluation import evaluate_baseline
from mot_pa2.types import Detection
from mot_pa2.synthetic import SyntheticConfig,generate_video,corrupt_detections

def test_empty_frame_clock():
    box=(10,10,20,20)
    det={1:[Detection(1,box)],3:[Detection(3,box)]}
    gt={1:[Detection(1,box,gt_id=1)],2:[Detection(2,box,gt_id=1)],3:[Detection(3,box,gt_id=1)]}
    result,pred,_=evaluate_baseline(det,gt,max_missed=0)
    assert list(pred)==[1,2,3]
    assert pred[2]==[]
    assert result["IDSW"]==1 and result["fragmentations"]==1

def test_exact_full_occlusion_and_easy_baseline():
    _,truth=generate_video(SyntheticConfig(objects=5,typical_speed=.15,occlusion_duration=8))
    absent=[f for f,items in enumerate(truth) if 1 not in {x.gt_id for x in items}]
    assert len(absent)==8
    assert absent==list(range(absent[0],absent[0]+8))
    _,truth=generate_video(SyntheticConfig(objects=5,typical_speed=.15,occlusion_duration=0))
    det=corrupt_detections(truth,0,0,0)
    metrics,_,_=evaluate_baseline(dict(enumerate(det)),dict(enumerate(truth)))
    assert metrics["IDF1"]>=.98
