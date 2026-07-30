#!/usr/bin/env python3

import VO
import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

# Mount both sequence directories under /dataset
app.mount("/dataset/KITTI_sequence_1", StaticFiles(directory="KITTI_sequence_1"), name="seq1")
app.mount("/dataset/KITTI_sequence_2", StaticFiles(directory="KITTI_sequence_2"), name="seq2")

# Global variables to handle process state
queue = multiprocessing.Queue()
vo_process = None

class StartRequest(BaseModel):
    dataset: str = "KITTI_sequence_2"

@app.get("/")
def get():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/start")
def start_vo(req: StartRequest):
    """Endpoint triggered by frontend to start or restart the VO process with a chosen dataset."""
    global vo_process
    
    # Terminate process if already running (allows dynamic restarts)
    if vo_process is not None and vo_process.is_alive():
        vo_process.terminate()
        vo_process.join()
    
    # Flush queue
    while not queue.empty():
        try:
            queue.get_nowait()
        except Exception:
            break

    print(f"Frontend requested pipeline start for [{req.dataset}]. Booting VO Process...")
    vo_process = multiprocessing.Process(target=run_vo_pipeline, args=(queue, req.dataset))
    vo_process.start()
    return {"status": "started", "dataset": req.dataset}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if not queue.empty():
                data = queue.get()
                await websocket.send_json(data)
            else:
                import asyncio
                await asyncio.sleep(0.01)
    except Exception as e:
        print(f"Client disconnected: {e}")

def run_vo_pipeline(q, dataset_name):
    print(f"VO Process Started. Loading Data from {dataset_name}...")
    vo = VO.VisualOdometry(dataset_name)
    
    print("Beginning Trajectory Estimation...")
    
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            
            # Push to the WebSocket Queue for the frontend
            q.put({
                "status": "running",
                "frame": i,
                "gt_pose": [gt_pose[0,3], 0, gt_pose[2,3]],
                "pred_pose": [cur_pose[0,3], 0, cur_pose[2,3]]
            })
            
    print("VO Pipeline Complete.")
    q.put({"status": "done"})

if __name__ == "__main__":
    print("Starting server... Open http://localhost:8000 in your browser.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
