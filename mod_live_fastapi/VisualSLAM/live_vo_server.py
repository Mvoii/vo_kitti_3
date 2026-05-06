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

app = FastAPI()

# Mount the KITTI directory so the frontend can request the images over HTTP
app.mount("/dataset", StaticFiles(directory="KITTI_sequence_2"), name="dataset")

# Global variables to handle the process state
queue = multiprocessing.Queue()
vo_process = None

@app.get("/")
def get():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/start")
def start_vo():
    """Endpoint triggered by the frontend button to start the VO process."""
    global vo_process
    
    if vo_process is not None and vo_process.is_alive():
        return {"status": "already running"}
    
    while not queue.empty():
        queue.get()

    print("Frontend requested pipeline start. Booting VO Process...")
    vo_process = multiprocessing.Process(target=run_vo_pipeline, args=(queue,))
    vo_process.start()
    return {"status": "started"}

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

# The VO Loop that runs in a separate process
def run_vo_pipeline(q):
    print("VO Process Started. Loading Data...")
    data_dir = 'KITTI_sequence_2'
    vo = VO.VisualOdometry(data_dir)
    
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
    # Tell the frontend the loop is finished
    q.put({"status": "done"})

if __name__ == "__main__":
    print("Starting server... Open http://localhost:8000 in your browser.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
