from metrics import evaluate_tracking,idf1,id_switches,detection_map

def record(track_id,gt_id):
    return dict(track_id=track_id,gt_id=gt_id,bbox=(gt_id*20,0,gt_id*20+10,10))

def test_perfect_identity_has_idf1_one():
    gt=[[record(1,1),record(2,2)] for _ in range(4)]
    assert idf1(gt,gt)==1
    assert id_switches(gt,gt)==0

def test_swap_counts_one_switch_per_identity():
    gt=[[record(1,1),record(2,2)] for _ in range(4)]
    pred=gt[:2]+[[record(2,1),record(1,2)] for _ in range(2)]
    assert id_switches(pred,gt)==2
    assert idf1(pred,gt)==.5

def test_track_split_has_different_effect_than_swap():
    gt=[[record(1,1),record(2,2)] for _ in range(4)]
    pred=gt[:2]+[[record(3,1),record(2,2)] for _ in range(2)]
    assert idf1(pred,gt)==.75
    assert id_switches(pred,gt)==1

def test_fragmentation_and_missing_frames():
    gt={1:[record(1,1)],2:[record(1,1)],3:[record(1,1)]}
    pred={1:[record(9,1)],3:[record(9,1)]}
    result=evaluate_tracking(pred,gt)
    assert result["fragmentations"]==1
    assert result["IDF1"]==.8

def test_identity_labels_are_not_used_to_match_predictions():
    gt=[[record(1,1)]]
    assert idf1([[dict(track_id=7,bbox=(20,0,30,10))]],gt)==1
    assert idf1([[dict(track_id=7,gt_id=1,bbox=(200,0,210,10))]],gt)==0

def test_global_assignment_large_case():
    gt=[[record(i,i) for i in range(25)] for _ in range(2)]
    pred=[[record(i+100,i) for i in range(25)] for _ in range(2)]
    assert idf1(pred,gt)==1

def test_ap_perfect_and_duplicate_false_positive():
    gt=[[record(1,1)]]
    assert detection_map(gt,gt)==1
    assert detection_map([[dict(bbox=(100,100,110,110),score=1)]],gt)==0
