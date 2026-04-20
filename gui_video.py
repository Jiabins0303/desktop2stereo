# gui_video.py - TensorRT Video Benchmark GUI
# A standalone GUI for benchmarking TensorRT depth models with video input

import os
import sys
import threading
import time
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Load settings from YAML
try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

# Default parameters (fallback if settings.yaml not found)
DEFAULTS = {
    "IPD": 0.064,
    "Depth Strength": 2.0,
    "Anti-aliasing": 1,
    "Foreground Scale": 0.5,
    "FP16": False,
    "Fill 16:9": True,
    "Show FPS": True,
    "Display Mode": "Full-SBS",
}


def load_settings():
    """Load settings from settings.yaml or use defaults."""
    settings = DEFAULTS.copy()
    if HAVE_YAML and os.path.exists("settings.yaml"):
        try:
            with open("settings.yaml", "r", encoding="utf-8") as f:
                yaml_settings = yaml.safe_load(f)
                if yaml_settings:
                    for key in DEFAULTS:
                        if key in yaml_settings:
                            settings[key] = yaml_settings[key]
        except Exception as e:
            print(f"[Warning] Failed to load settings.yaml: {e}")
    return settings


# Load settings
SETTINGS = load_settings()
IPD = SETTINGS["IPD"]
DEPTH_STRENGTH = SETTINGS["Depth Strength"]
AA_STRENGTH = SETTINGS["Anti-aliasing"]
FOREGROUND_SCALE = SETTINGS["Foreground Scale"]
FP16 = SETTINGS["FP16"]
FILL_16_9 = SETTINGS["Fill 16:9"]
SHOW_FPS = SETTINGS["Show FPS"]
DISPLAY_MODE = SETTINGS["Display Mode"]

# Determine device and dtype
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if FP16 and torch.cuda.is_available() else torch.float32

# ImageNet normalization constants (will be initialized when CUDA is available)
IMAGENET_MEAN = None
IMAGENET_STD = None

def init_normalization_constants():
    """Initialize ImageNet normalization constants on the correct device."""
    global IMAGENET_MEAN, IMAGENET_STD
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE, dtype=DTYPE).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE, dtype=DTYPE).view(1, 3, 1, 1)


class TensorRTEngine:
    """TensorRT Engine wrapper for direct .trt file loading."""
    
    def __init__(self, engine_path, device, dtype):
        """Initialize TensorRT engine from .trt file."""
        self.device = device
        self.dtype = dtype
        self.input_shape = None
        
        try:
            import tensorrt as trt
            
            # Load TensorRT engine
            print(f"[TensorRT] Loading engine: {engine_path}")
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            
            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
            
            # Get binding information
            self.input_binding_indices = []
            self.output_binding_indices = []
            
            for binding in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(binding)
                mode = self.engine.get_tensor_mode(name)
                
                if mode == trt.TensorIOMode.INPUT:
                    self.input_binding_indices.append(binding)
                else:
                    self.output_binding_indices.append(binding)
            
            # Store input name and shape
            if self.input_binding_indices:
                first_input_binding = self.input_binding_indices[0]
                self.input_name = self.engine.get_tensor_name(first_input_binding)
                engine_input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
                if all(dim > 0 for dim in engine_input_shape):
                    self.input_shape = engine_input_shape
                    print(f"[TensorRT] Engine has fixed input shape: {self.input_shape}")
            else:
                raise RuntimeError("No input bindings found in TensorRT engine")
            
            # Store output name
            if self.output_binding_indices:
                first_output_binding = self.output_binding_indices[0]
                self.output_name = self.engine.get_tensor_name(first_output_binding)
            else:
                raise RuntimeError("No output bindings found in TensorRT engine")
            
            # Pre-allocate output shapes
            self.output_shapes = {}
            for binding in self.output_binding_indices:
                name = self.engine.get_tensor_name(binding)
                self.output_shapes[name] = self.engine.get_tensor_shape(name)
            
            print(f"[TensorRT] Engine loaded successfully")
            
        except ImportError:
            raise ImportError("TensorRT not available. Please install TensorRT.")
    
    def get_fixed_input_size(self):
        """Return the fixed input size (H, W) if available."""
        if self.input_shape is not None and len(self.input_shape) == 4:
            return (self.input_shape[2], self.input_shape[3])
        return None
    
    def __call__(self, tensor):
        """Execute inference."""
        import tensorrt as trt
        
        input_shape = tuple(tensor.shape)
        name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(name, input_shape)
        
        outputs = {}
        bindings = [None] * self.engine.num_io_tensors
        bindings[0] = tensor.data_ptr()
        
        for i, binding in enumerate(self.output_binding_indices, 1):
            name = self.engine.get_tensor_name(binding)
            dims = self.context.get_tensor_shape(name)
            shape_tuple = tuple(dims)
            
            # Get the actual output dtype from the engine
            # INT8 models still output float32
            trt_dtype = self.engine.get_tensor_dtype(name)
            if trt_dtype == trt.float32:
                torch_dtype = torch.float32
            elif trt_dtype == trt.float16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32  # Default to float32 for INT8 outputs
            
            output = torch.empty(shape_tuple, device=self.device, dtype=torch_dtype)
            outputs[name] = output
            bindings[binding] = output.data_ptr()
        
        self.context.execute_v2(bindings=bindings)
        return outputs[self.output_name]


def preprocess_frame(frame, input_size, device, dtype):
    """
    Preprocess frame for depth model inference.
    
    Args:
        frame: BGR numpy array from OpenCV
        input_size: (H, W) tuple for model input
        device: torch device
        dtype: torch dtype
    
    Returns:
        Preprocessed tensor ready for inference
    """
    global IMAGENET_MEAN, IMAGENET_STD
    
    # Initialize constants if needed
    if IMAGENET_MEAN is None:
        init_normalization_constants()
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and move to GPU
    tensor = torch.from_numpy(rgb).to(device=device, dtype=dtype)
    
    # Reshape to NCHW format
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    # Normalize to [0, 1]
    tensor = tensor / 255.0
    
    # Resize to model input size
    tensor = F.interpolate(tensor, size=input_size, mode='bilinear', align_corners=False)
    
    # Apply ImageNet normalization
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    
    # Ensure contiguous memory layout
    return tensor.contiguous()


def normalize_tensor(tensor: torch.Tensor):
    """DirectML-safe normalization to [0,1], ignoring NaNs."""
    mask = ~torch.isnan(tensor)
    if not mask.any():
        return torch.zeros_like(tensor)
    valid = tensor[mask]
    min_val = valid.min()
    max_val = valid.max()
    denom = max_val - min_val
    if denom == 0:
        return torch.zeros_like(tensor)
    out = (tensor - min_val) / denom
    out[~mask] = 0.0
    return out


def apply_gamma(depth, gamma=1.2):
    """Apply gamma correction to depth map."""
    return torch.pow(depth.clamp(min=1e-8), gamma)


def apply_contrast(depth, factor=1.2):
    """Apply contrast adjustment to depth map."""
    mean = depth.mean(dim=(-2, -1), keepdim=True)
    return torch.clamp((depth - mean) * factor + mean, 0, 1)


def apply_foreground_scale(depth: torch.Tensor, scale: float, mid: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Scale depth contrast for foreground/background separation."""
    if not (-1.0 + 1e-12 < scale):
        raise ValueError("scale must be greater than -1.0")
    
    d = depth.clamp(0.0, 1.0)
    if abs(scale) < eps:
        return d
    
    exponent = 1.0 / (1.0 + scale)
    dist = d - mid
    out = mid + torch.sign(dist) * torch.pow(torch.abs(dist).clamp(min=1e-8), exponent)
    return out.clamp(0.0, 1.0)


def anti_alias(depth: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Apply anti-aliasing to reduce jagged edges in depth maps."""
    if strength <= 0:
        return depth
    
    original_shape = depth.shape
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)
    
    # Create Gaussian kernel
    kernel_size = max(3, int(strength * 2) * 2 + 1)
    sigma = strength
    
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, device=depth.device, dtype=depth.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D kernel
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution with padding
    padding = kernel_size // 2
    depth = F.conv2d(depth, kernel_2d, padding=padding)
    
    # Restore original shape
    if len(original_shape) == 2:
        depth = depth.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        depth = depth.squeeze(1)
    
    return depth


def postprocess_depth(depth_output, original_size, foreground_scale=0.5, aa_strength=1):
    """
    Post-process depth output to match depth.py processing.
    
    Args:
        depth_output: Raw depth tensor from model
        original_size: (H, W) tuple for output size
        foreground_scale: Foreground scaling factor
        aa_strength: Anti-aliasing strength
    
    Returns:
        Normalized depth map tensor
    """
    # Convert to float32 for processing (important for INT8 models)
    depth = depth_output.float()
    
    # Normalize and squeeze
    depth = normalize_tensor(depth).squeeze()
    
    # Ensure 2D tensor
    while depth.dim() > 2:
        depth = depth.squeeze(0)
    
    # Apply gamma correction
    depth = apply_gamma(depth)
    
    # Apply contrast
    depth = apply_contrast(depth)
    
    # Apply foreground scale
    depth = apply_foreground_scale(depth, scale=foreground_scale)
    
    # Apply anti-aliasing
    depth = anti_alias(depth, strength=aa_strength)
    
    # Final normalization
    depth = normalize_tensor(depth).squeeze()
    
    # Resize to original size
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    
    depth = F.interpolate(depth, size=original_size, mode='bilinear', align_corners=False)
    depth = depth.squeeze()
    
    return depth


def video_inference_thread(video_path, engine, input_size, original_size, frame_queue, stop_event):
    """
    Background thread for video reading and inference.
    Pushes (rgb_frame, depth) tuples to frame_queue.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[Error] Failed to open video file")
        return
    
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Preprocess
            input_tensor = preprocess_frame(frame, input_size, DEVICE, DTYPE)
            
            # Inference
            with torch.no_grad():
                depth_output = engine(input_tensor)
            
            # Post-process
            depth = postprocess_depth(depth_output, original_size, FOREGROUND_SCALE, AA_STRENGTH)
            
            # Convert frame to RGB for viewer
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Put in queue (non-blocking, drop old frames if queue is full)
            try:
                # Clear old frame if queue has one
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait((rgb_frame, depth))
            except queue.Full:
                pass
    
    finally:
        cap.release()
        print("[Inference Thread] Finished")


def run_viewer_loop(trt_path, video_path):
    """
    Run the main viewer loop (must be called from main thread).
    Returns average FPS when done.
    """
    import glfw
    from viewer import StereoWindow
    
    # Load TensorRT engine
    print(f"[TensorRT] Loading engine: {trt_path}")
    engine = TensorRTEngine(trt_path, DEVICE, DTYPE)
    input_size = engine.get_fixed_input_size()
    
    if input_size is None:
        input_size = (518, 518)
        print(f"[Warning] Dynamic input detected, using default size: {input_size}")
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video file")
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    original_size = (video_height, video_width)
    
    # Create frame queue
    frame_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    
    # Start inference thread
    inference_thread = threading.Thread(
        target=video_inference_thread,
        args=(video_path, engine, input_size, original_size, frame_queue, stop_event),
        daemon=True
    )
    inference_thread.start()
    
    # Create viewer (in main thread)
    viewer = StereoWindow(
        ipd=IPD,
        depth_ratio=DEPTH_STRENGTH,
        display_mode=DISPLAY_MODE,
        fill_16_9=FILL_16_9,
        show_fps=SHOW_FPS,
        use_3d=False,
        fix_aspect=False,
        stream_mode=None,
        frame_size=(video_width, video_height)
    )
    
    print("[Viewer] Starting viewer loop...")
    print("[Viewer] Controls: Tab=Switch View | Up/Down=Depth | Esc=Close")
    
    # Main viewer loop
    frame_count = 0
    start_time = time.perf_counter()
    
    try:
        while not glfw.window_should_close(viewer.window):
            # Get latest frame from queue
            try:
                rgb_frame, depth = frame_queue.get(timeout=0.1)
                viewer.update_frame(rgb_frame, depth)
                frame_count += 1
            except queue.Empty:
                pass
            
            # Render
            viewer.render()
            glfw.swap_buffers(viewer.window)
            glfw.poll_events()
    
    finally:
        # Stop inference thread
        stop_event.set()
        inference_thread.join(timeout=2.0)
        
        # Calculate average FPS
        elapsed = time.perf_counter() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"[Benchmark] Processed {frame_count} frames in {elapsed:.2f}s (avg {avg_fps:.1f} FPS)")
        
        # Cleanup viewer
        glfw.destroy_window(viewer.window)
        glfw.terminate()
        
        return avg_fps


class VideoTRTBenchmarkGUI:
    """Main GUI class for TensorRT video benchmarking."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("TensorRT Video Benchmark")
        self.root.geometry("550x320")
        self.root.resizable(False, False)
        
        # State variables
        self.trt_path = tk.StringVar(value="")
        self.video_path = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="Ready")
        
        # Create GUI
        self._create_widgets()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Row 0: TRT Model Selection
        ttk.Label(main_frame, text="TensorRT Model (.trt):").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        trt_frame = ttk.Frame(main_frame)
        trt_frame.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=5)
        
        self.trt_entry = ttk.Entry(trt_frame, textvariable=self.trt_path, width=40)
        self.trt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(trt_frame, text="Browse...", command=self._browse_trt).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Row 1: Video Selection
        ttk.Label(main_frame, text="Video File (.mp4):").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=5)
        
        self.video_entry = ttk.Entry(video_frame, textvariable=self.video_path, width=40)
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(video_frame, text="Browse...", command=self._browse_video).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Row 2: Info labels
        info_frame = ttk.LabelFrame(main_frame, text="Info", padding="5")
        info_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        self.model_info_label = ttk.Label(info_frame, text="Model: Not selected")
        self.model_info_label.pack(anchor=tk.W)
        
        self.video_info_label = ttk.Label(info_frame, text="Video: Not selected")
        self.video_info_label.pack(anchor=tk.W)
        
        # Row 3: Controls
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Benchmark", command=self._start_benchmark, width=20)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        # Row 4: Status
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_text)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Row 5: Instructions
        instructions_text = (
            "Instructions:\n"
            "1. Select a TensorRT model (.trt file)\n"
            "2. Select a video file (.mp4)\n"
            "3. Click 'Start Benchmark' to open the viewer\n"
            "Controls: Tab=Switch View | Up/Down=Adjust Depth | Esc=Close"
        )
        instructions = ttk.Label(main_frame, text=instructions_text, font=("TkDefaultFont", 8), justify=tk.LEFT)
        instructions.grid(row=5, column=0, columnspan=3, pady=5, sticky=tk.W)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
    
    def _browse_trt(self):
        """Browse for TRT model file."""
        filepath = filedialog.askopenfilename(
            title="Select TensorRT Model",
            filetypes=[("TensorRT Engine", "*.trt"), ("All Files", "*.*")],
            initialdir="models"
        )
        if filepath:
            self.trt_path.set(filepath)
            filename = os.path.basename(filepath)
            self.model_info_label.config(text=f"Model: {filename}")
    
    def _browse_video(self):
        """Browse for video file."""
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 Video", "*.mp4"), ("All Video", "*.mp4;*.avi;*.mkv;*.mov"), ("All Files", "*.*")]
        )
        if filepath:
            self.video_path.set(filepath)
            # Get video info
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                filename = os.path.basename(filepath)
                self.video_info_label.config(
                    text=f"Video: {filename} ({width}x{height}, {fps:.1f}fps, {duration:.1f}s)"
                )
            else:
                self.video_info_label.config(text="Video: Failed to open")
    
    def _start_benchmark(self):
        """Start the benchmark (launches viewer)."""
        # Validate inputs
        trt_path = self.trt_path.get()
        video_path = self.video_path.get()
        
        if not trt_path or not os.path.exists(trt_path):
            messagebox.showerror("Error", "Please select a valid TensorRT model file.")
            return
        
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return
        
        # Hide GUI window
        self.root.withdraw()
        self.status_text.set("Running benchmark...")
        
        try:
            # Run viewer loop (blocking, in main thread)
            avg_fps = run_viewer_loop(trt_path, video_path)
            self.status_text.set(f"Benchmark complete. Avg FPS: {avg_fps:.1f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_text.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            # Show GUI window again
            self.root.deiconify()
    
    def _on_closing(self):
        """Handle window close event."""
        self.root.destroy()


def main():
    """Main entry point."""
    # Check for CUDA
    if not torch.cuda.is_available():
        print("[Warning] CUDA not available. TensorRT requires CUDA.")
        messagebox.showwarning("Warning", "CUDA not available. TensorRT requires CUDA for inference.")
    
    # Create and run GUI
    root = tk.Tk()
    app = VideoTRTBenchmarkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
