import multiprocessing as mp
import time
import signal
import sys
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from queue import Empty, Full
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/pipeline.log')
    ]
)

class QueueManager:
    _instance = None
    _queues: Dict[str, mp.Queue] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueueManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def create_queue(cls, name: str, maxsize: int = 0) -> mp.Queue:
        """Create a new multiprocessing queue"""
        if name not in cls._queues:
            cls._queues[name] = mp.Queue(maxsize=maxsize)
        return cls._queues[name]

    @classmethod
    def get_queue(cls, name: str) -> Optional[mp.Queue]:
        """Get an existing queue by name"""
        return cls._queues.get(name)

    @classmethod
    def cleanup(cls):
        """Close all queues"""
        for name, q in cls._queues.items():
            try:
                q.close()
                q.cancel_join_thread()
            except Exception as e:
                logging.error(f"Error closing queue {name}: {e}")
        cls._queues.clear()

class ProcessBase(mp.Process, ABC):
    """Base class for all pipeline processes"""
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name=name)
        self.config = config
        self.should_exit = mp.Event()
        self.logger = logging.getLogger(name)
        self._input_queues = {}
        self._output_queues = {}

    def register_input_queue(self, name: str, queue: mp.Queue):
        self._input_queues[name] = queue

    def register_output_queue(self, name: str, queue: mp.Queue):
        self._output_queues[name] = queue

    def stop(self):
        """Signal the process to stop"""
        self.should_exit.set()

    def run(self):
        """Main process loop"""
        self.logger.info(f"Process {self.name} started (PID: {self.pid})")
        
        try:
            self.setup()
            
            while not self.should_exit.is_set():
                try:
                    self.loop()
                except Exception as e:
                    self.logger.error(f"Error in process loop: {e}")
                    self.logger.error(traceback.format_exc())
                    # Optional: decide whether to break or continue based on severity
                    time.sleep(0.1)  # Prevent tight loop on error
                    
        except Exception as e:
            self.logger.critical(f"Fatal error in process {self.name}: {e}")
            self.logger.critical(traceback.format_exc())
        finally:
            self.cleanup()
            self.logger.info(f"Process {self.name} stopped")

    def setup(self):
        """Called once before the main loop"""
        pass

    @abstractmethod
    def loop(self):
        """Main processing logic, called repeatedly"""
        pass

    def cleanup(self):
        """Called on exit"""
        pass

class ProcessManager:
    def __init__(self):
        self.processes: Dict[str, ProcessBase] = {}
        self.logger = logging.getLogger("ProcessManager")
        self.running = False

    def add_process(self, process: ProcessBase):
        self.processes[process.name] = process

    def start_all(self):
        """Start all registered processes"""
        self.logger.info("Starting all processes...")
        self.running = True
        
        # Set start method to spawn (required for CUDA/Windows)
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        for name, process in self.processes.items():
            process.start()
            self.logger.info(f"Started {name}")

    def stop_all(self):
        """Stop all processes gracefully"""
        self.logger.info("Stopping all processes...")
        self.running = False
        
        # Signal all processes to stop
        for process in self.processes.values():
            process.stop()

        # Wait for processes to join
        for name, process in self.processes.items():
            process.join(timeout=5.0)
            if process.is_alive():
                self.logger.warning(f"Process {name} did not stop gracefully, terminating...")
                process.terminate()
        
        QueueManager.cleanup()
        self.logger.info("All processes stopped")

    def monitor(self):
        """Monitor process health"""
        while self.running:
            for name, process in self.processes.items():
                if not process.is_alive():
                    self.logger.error(f"Process {name} died unexpectedly!")
                    # TODO: Implement restart logic
            time.sleep(1.0)

# Global accessor
def get_process_manager():
    return ProcessManager()
