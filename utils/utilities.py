import threading
import cv2
import queue

class AsyncVideoWriter(threading.Thread):
    def __init__(self, filename, fps=10.0, max_queue=130):
        super().__init__()
        self.filename = filename
        self.fps = fps
        self.max_queue = max_queue
        self.q = queue.Queue(maxsize=max_queue)
        self.writer = None
        self.running = True

    def enqueue(self, frame):
        if self.running and not self.q.full():
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass

    def run(self):
        while self.running or not self.q.empty():
            try:
                frame = self.q.get(timeout=1.0)
                if self.writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (w, h))
                self.writer.write(frame)
                self.q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncVideoWriter] Error: {e}")

        if self.writer:
            self.writer.release()

    def stop(self):
        self.running = False
        self.join()
