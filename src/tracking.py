"""Own association and persistent recurrent state with an explicit frame clock."""
import torch
from .association import IoUTracker
from .temporal import box_to_features,features_to_box

def records(tracks):
    return [dict(track_id=t.track_id,bbox=t.bbox,missed=t.missed) for t in tracks]

def run_baseline(detections_by_frame,matcher="greedy",iou_threshold=.3,max_missed=3):
    tracker=IoUTracker(iou_threshold,max_missed,matcher)
    return {f:records(tracker.update(detections_by_frame.get(f,[])))
            for f in range(min(detections_by_frame),max(detections_by_frame)+1)} if detections_by_frame else {}

class TemporalTracker:
    def __init__(self,model,image_size=(128,128),iou_threshold=.3,max_missed=3):
        self.model=model.eval()
        self.image_size=image_size
        self.tracker=IoUTracker(iou_threshold,max_missed)
        self.states,self.next_boxes={},{}

    @torch.no_grad()
    def update(self,detections):
        predicted=dict(self.next_boxes)
        for track in self.tracker.tracks: track.bbox=self.next_boxes[track.track_id]
        tracks=self.tracker.update(detections)
        alive={t.track_id for t in tracks}
        self.states={k:v for k,v in self.states.items() if k in alive}
        self.next_boxes={}
        for track in tracks:
            observation=box_to_features(track.bbox,self.image_size)[None]
            prediction,state=self.model.step(observation,self.states.get(track.track_id))
            self.states[track.track_id]=state
            self.next_boxes[track.track_id]=features_to_box(prediction[0].tolist(),self.image_size)
        return records(tracks),predicted

def run_temporal(detections_by_frame,model,iou_threshold=.3,max_missed=3,image_size=(128,128),return_debug=False):
    tracker=TemporalTracker(model,image_size,iou_threshold,max_missed)
    predictions,debug={},{}
    if detections_by_frame:
        for frame in range(min(detections_by_frame),max(detections_by_frame)+1):
            predictions[frame],debug[frame]=tracker.update(detections_by_frame.get(frame,[]))
    return (predictions,debug) if return_debug else predictions
