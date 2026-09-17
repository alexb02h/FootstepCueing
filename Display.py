import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import queue,sys, threading, cv2, tkinter as tk 
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import ttk
from PIL import Image, ImageTk
from DetectWalkers import main_process
import numpy as np
import mymodule #type: ignore
from ultralytics import YOLO
import concurrent.futures
from io import StringIO

def init_worker() : 
    global pose_model, vehicle_model
    pose_model = YOLO("YoloModels/yolo11n-pose.pt")
    vehicle_model = YOLO("YoloModels/yolo11n.pt")   

def worker_process(clip_item) : 
    idx, clip_path = clip_item
    global pose_model, vehicle_model
    
    temp_path = f"media/Metadata/temp_annotated_{idx}.mp4"
    
    stdout_capture = StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture
    
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps) : fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    clip_writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    cue_lines = []
    
    class CueCapture : 
        def write(self, text) : cue_lines.append(text)
        
    clip_cue = CueCapture()
    
    try : main_process(clip_path, pose_model, vehicle_model, frame_callback=None, out_writer=clip_writer, cue_file=clip_cue)
    finally : 
        clip_writer.release()
        sys.stdout = old_stdout
        
    captured_logs= stdout_capture.getvalue()
    return idx, temp_path, cue_lines, captured_logs
    
pose_model = YOLO("YoloModels/yolo11n-pose.pt")
vehicle_model = YOLO("YoloModels/yolo11n.pt")   

class Redirection : 
    def __init__(self, text_queue) : self.text_queue = text_queue
    
    def write(self, string) : self.text_queue.put(string)
    
    def flush(self) : pass

root = TkinterDnD.Tk()
root.title("Test")
root.geometry("650x650")
root.configure(bg="#1e1e1e")

msg_queue = queue.Queue()

label = tk.Label(root, text="Hello, Loser!", font=("Arial",16))
label.pack(pady=10)

preview = tk.Frame(root, width=320, height=220, bg="#2d2d2d")
preview.pack(pady=10)
preview.pack_propagate(False)

preview_label = tk.Label(preview, text="No File Loaded", bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 12))
preview_label.pack(expand=True, fill="both")

log_label = tk.Label(root, text="Process Console Output : ", font=("Arial", 11, "bold"), fg="#a0a0a0", bg="#1e1e1e")
log_label.pack(anchor="w", padx=30, pady=(10, 2))

log_frame = tk.Frame(root, bg="#1e1e1e")
log_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))

log_text = tk.Text(log_frame, bg="#0d1117", fg="#4ade80", font=("Courier", 9), relief="flat", wrap="word")
log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
log_text.configure(yscrollcommand=log_scroll.set)

log_scroll.pack(side="right", fill="y")
log_text.pack(side="left", fill="both", expand=True)

def update_log() : 
    while not msg_queue.empty() :
        msg = msg_queue.get_nowait()
        log_text.insert(tk.END, msg)
        log_text.see(tk.END)
    root.after(100, update_log)
    
root.after(100, update_log)
current_image = None

def render(cv2_frame)  :
    global current_frame
    h, w = cv2_frame.shape[:2]
    target_w, target_h = 480, 270
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(cv2_frame, (nw, nh))
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(rgb_frame)
    img_tk = ImageTk.PhotoImage(image=img)
    
    def update_widget() : 
        global current_frame
        current_frame = img_tk
        preview_label.config(image=current_frame, text="", bg="#000000")
    
    root.after(0, update_widget)

def get_dur(file_path) : 
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened() : return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0 : return frame_count / fps
    return 0.0

def stitch(temp_path, output_path) :
    if not temp_path : return
    
    cap = cv2.VideoCapture(temp_path[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for path in temp_path : 
        cap = cv2.VideoCapture(path)
        while True : 
            ret, frame = cap.read()
            if not ret : break
            out.write(frame)
        cap.release()
        os.remove(path)
    out.release()
    
def run_back(file_path, pose_model, vehicle_model) : 
    old_stdout = sys.stdout
    sys.stdout = Redirection(msg_queue)
    filename = os.path.basename(file_path)
    
    try : 
        duration = get_dur(file_path)
        if duration > 60 : 
            os.makedirs("media/rgbframes", exist_ok=True)
            os.makedirs("media/grayframes", exist_ok=True)
            os.makedirs("media/Metadata", exist_ok=True)
            track_counter = [1]
            mymodule.PreProcess(file_path)
            
            valid_ext = (".mp4", ".avi", ".mov", ".mkv")
            clip_files = sorted([os.path.join("media/rgbframes", f) for f in os.listdir("media/rgbframes") if f.lower().endswith(valid_ext)])
            
            if not clip_files : raise RuntimeError("No Clips were generated")
            
            max_workers = min(4, os.cpu_count() or 1)
            indexed_clips = list(enumerate(clip_files, start=1))
            results = []
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor : 
                futures = [executor.submit(worker_process, item) for item in indexed_clips]
                for future in concurrent.futures.as_completed(futures) : 
                    idx, temp_path, cue_lines, logs = future.result()
                    results.append((idx, temp_path, cue_lines))
                    if logs : print(f"[Clip {idx} Logs]:\n{logs}")
                    print(f"Finished processing clip batch {idx}/{len(clip_files)}")
                    
            results.sort(key=lambda x: x[0])

            print("\nStitching processed clips into final output...")
            temp_video_files = [res[1] for res in results]
            stitch(temp_video_files, "annotated.mp4")

            with open("Walkers.cue", "w") as cue_file:
                cue_file.write('TITLE "Movement and Character Motion Markers"\n')
                cue_file.write(f'FILE "{os.path.basename(file_path)}" MP4\n')
                for res in results:
                    for line in res[2]:
                        cue_file.write(line)
            
        else : main_process(file_path, pose_model, vehicle_model, frame_callback=render)
        
        cue_summary = ""
        if os.path.exists("Walkers.cue") :
            with open("Walkers.cue", "r") as f :
                line = f.readlines()
                cue_summary = f"\nGenerated {len(line)} cue marker lines."
                
        print(f"\n=== Finished Processing Successfully! ===")
        
        root.after(0, lambda: label.config(text=f"Completed: {filename}", fg="#4ade80"))
        root.after(0, lambda: preview_label.config(
            text=f"PROCESSING COMPLETE\n\nOutput Video: annotated.mp4\nOutput Markers: Walkers.cue{cue_summary}",
            bg="#1e293b", fg="#ffffff", font=("Arial", 11, "bold")
        ))
    except Exception as e :
        error_message = str(e)
        print(f"\n[ERROR]: {e}")
        root.after(0, lambda: label.config(text="Processing Failed", fg="#f87171"))
        root.after(0, lambda msg=error_message: preview_label.config(text=f"Error during execution:\n{msg}", bg="#2d2d2d", fg="#f87171", font=("Arial", 11)))
    finally : sys.stdout = old_stdout
    
def handle(event) :
    global current
    file_path = event.data.strip("{}")
    
    if not os.path.isfile(file_path) : return
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    log_text.delete("1.0", tk.END)
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"] : 
        try : 
            img = Image.open(file_path)
            img.thumbnail((300, 200))
            
            current = ImageTk.PhotoImage(img)
            preview_label.config(image=current, text="", bg="#1e1e1e")
            label.config(text=f"Loaded: {os.path.basename(file_path)}")
        except Exception as e : preview_label.config(image="", text="Failed to render image preview", bg="#2d2d2d")
    elif ext in[".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"] : 
       label.config(text=f"Processing Video : {filename}")
       preview_label.config(image="", text=f"\n\nRunning main_process on : \n {filename}\n", bg="#1e293b", fg="#60a5fa", font=("Arial", 11, "bold")) 
       threading.Thread(target=run_back, args=(file_path, pose_model, vehicle_model), daemon=True).start()
    else:
        file_size = round(os.path.getsize(file_path) / 1024, 1)
        preview_label.config(image="", text=f"\n\nName: {filename}\nType : {ext.upper() if ext else 'Unknown'}\nSize : {file_size}kb", bg="#1e293b", fg="#ffffff", font=("Courier", 11, "bold"))
        label.config(text=f"Loaded : {filename}")

if __name__ == "__main__" :
    root.drop_target_register(DND_FILES)
    root.dnd_bind("<<Drop>>", handle)
    root.mainloop()