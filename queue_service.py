from queue import Queue
from threading import Thread
from typing import Dict, Any, Callable
import time
import logging

class BackgroundQueue:
    """Background queue for handling API submissions"""
    
    def __init__(self, process_func: Callable):
        self.queue: Queue = Queue()
        self.process_func = process_func
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()
    
    def add_task(self, task: Dict[str, Any]):
        """Add a task to the queue"""
        self.queue.put(task)
    
    def _process_queue(self):
        """Process tasks in the background"""
        while True:
            try:
                if not self.queue.empty():
                    task = self.queue.get()
                    logging.debug(f"Processing task: {task}")
                    self.process_func(task)
                    self.queue.task_done()
                else:
                    time.sleep(0.1)  # Prevent CPU spinning
            except Exception as e:
                logging.error(f"Error processing queue task: {str(e)}")
