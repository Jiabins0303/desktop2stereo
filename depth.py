# depth.py
import torch
torch.set_num_threads(1)
from utils import DEVICE_ID, MODEL_ID, CACHE_PATH, FP16, DEPTH_RESOLUTION, AA_STRENGTH, FOREGROUND_SCALE, USE_TORCH_COMPILE, USE_TENSORRT, RECOMPILE_TRT, FILL_16_9, OS_NAME, is_onnx_model, USE_ONNX, ONNX_MODEL_PATH
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
import cv2
import os, warnings

# ONNX Runtime for direct ONNX model inference
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("[Warning] onnxruntime not available. ONNX models will not be supported.")

# Initialize DirectML Device
def get_device(index=0):
    try:
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device(index), f"Using DirectML device: {torch_directml.device_name(index)}"
        except ImportError:
            pass
        if torch.backends.mps.is_available() and index==0:
            return torch.device("mps"), "Using Apple Silicon (MPS) device"
        if torch.cuda.is_available():
            return torch.device("cuda"), f"Using CUDA device: {torch.cuda.get_device_name(index)}"
        else:
            return torch.device("cpu"), "Using CPU device"
    except:
        return torch.device("cpu"), "Using CPU device"
    
DEVICE, DEVICE_INFO = get_device(DEVICE_ID)
print(DEVICE_INFO)
print(f"Model: {MODEL_ID}")

IS_CUDA = "CUDA" in DEVICE_INFO
IS_NVIDIA = "CUDA" in DEVICE_INFO and "NVIDIA" in DEVICE_INFO
IS_AMD_ROCM = "CUDA" in DEVICE_INFO and "AMD" in DEVICE_INFO
IS_DIRECTML = "DirectML" in DEVICE_INFO
IS_MPS = "MPS" in DEVICE_INFO

# check if it is metric model
def is_metric():
    if 'metric'  in MODEL_ID.lower() or 'kitti'  in MODEL_ID.lower() or 'nyu' in MODEL_ID.lower() or 'depth-ai' in MODEL_ID.lower() or 'da3' in MODEL_ID.lower():
        return True
    else:
        return False

# Optimization for CUDA
if IS_NVIDIA:
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable TF32 matrix multiplication for better performance
    torch.set_float32_matmul_precision('high')
    # Enable math attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] ="1" # Debug for torch.compile
    if USE_TORCH_COMPILE:
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"torch\._inductor\.lowering"
        )
    
elif IS_AMD_ROCM:
    torch.backends.cudnn.enabled = False # Add for AMD ROCm
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # Add for AMD ROCm7
    # Enable TF32 for matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = True
    # Enable math attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    if OS_NAME != "Linux":
        USE_TORCH_COMPILE = False  # Disable torch.compile for AMD ROCm7 due to current issues

# Model configuration
DTYPE = torch.float16 if FP16 else torch.float32
# Folder to store compiled model / cache 
MODEL_FOLDER = os.path.join(CACHE_PATH, "models--"+MODEL_ID.replace("/", "--"))
# Load depth model - either original or ONNX
DTYPE_INFO = "fp16" if FP16 else "fp32"
ONNX_PATH = os.path.join(MODEL_FOLDER, f"model_{DTYPE_INFO}_{DEPTH_RESOLUTION}.onnx")
TRT_PATH = os.path.join(MODEL_FOLDER, f"model_{DTYPE_INFO}_{DEPTH_RESOLUTION}.trt")


# Single character digits and letters for "FPS: XX.X"
font_dict = {
    "0": ["111","101","101","101","111"],
    "1": ["010","110","010","010","111"],
    "2": ["111","001","111","100","111"],
    "3": ["111","001","111","001","111"],
    "4": ["101","101","111","001","001"],
    "5": ["111","100","111","001","111"],
    "6": ["111","100","111","101","111"],
    "7": ["111","001","010","100","100"],
    "8": ["111","101","111","101","111"],
    "9": ["111","101","111","001","111"],
    "F": ["111","100","110","100","100"],
    "P": ["110","101","110","100","100"],
    "S": ["111","100","111","001","111"],
    ":": ["000","010","000","010","000"],
    ".": ["000","000","000","000","010"],  # for decimal point
    " ": ["000","000","000","000","000"],
}

# Post-processing functions
def apply_foreground_scale(depth: torch.Tensor, scale: float, mid: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    Scale depth contrast so that:
      - depth in [0,1], where 0 = background (far), 1 = foreground (near)
      - scale > 0 : increase separation (foreground closer -> values move toward 1, background farther -> values move toward 0)
      - scale < 0 : reduce separation (flatten)
      - scale = 0 : identity

    Args:
        depth: torch.Tensor shape (..., 1) or (...), values in [0,1]
        scale: float, must be > -1.0 (we avoid scale == -1 which would divide by zero)
        mid: midpoint for separation (default 0.5)
        eps: small eps to avoid numerical issues

    Returns:
        Tensor same shape as depth, clamped to [0,1].
    """
    if not (-1.0 + 1e-12 < scale):  # avoid scale <= -1
        raise ValueError("scale must be greater than -1.0")

    d = depth.clamp(0.0, 1.0)
    if abs(scale) < eps:
        return d

    exponent = 1.0 / (1.0 + scale)  # >1 if scale<0 (flatten), <1 if scale>0 (exaggerate)
    dist = d - mid
    out = mid + torch.sign(dist) * torch.pow(torch.abs(dist), exponent)
    return out.clamp(0.0, 1.0)
       
def anti_alias(depth: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """
    Apply anti-aliasing to reduce jagged edges in depth maps.
    
    Args:
        depth (torch.Tensor): Normalized depth map tensor [H,W] or [B,1,H,W] with values in [0,1].
        strength (float): Blur strength; higher = smoother edges. Recommended range [0.5, 2.0].
    
    Returns:
        torch.Tensor: Smoothed depth map with same shape.
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)  # [B,1,H,W]

    # Kernel size scales with strength
    k = int(3 * strength) | 1  # force odd number
    if k < 3:
        return depth.squeeze()

    # Gaussian blur kernel
    sigma = 0.5 * strength
    coords = torch.arange(k, device=depth.device, dtype=depth.dtype) - k // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss /= gauss.sum()

    # Separable convolution (X then Y)
    depth = F.conv2d(depth, gauss.view(1,1,1,-1), padding=(0, k//2), groups=1)
    depth = F.conv2d(depth, gauss.view(1,1,-1,1), padding=(k//2, 0), groups=1)

    return depth.squeeze()

def process_tensor(img_rgb: np.ndarray, height) -> torch.Tensor:
    """
    Convert BGR/UMat numpy image to normalized GPU tensor in model dtype.
    Keeps transfers efficient and uses non_blocking for pinned tensors where possible.
    """
    if isinstance(img_rgb, cv2.UMat):
        img_rgb = img_rgb.get()

    h0, w0 = img_rgb.shape[:2]
    if height < h0:
        width = int(img_rgb.shape[1] / h0 * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)

    # Ensure contiguous numpy array (uint8)
    np_img = np.ascontiguousarray(img_rgb)
    # convert to torch tensor on CPU (uint8) then float on device to avoid double copy
    t_cpu = torch.from_numpy(np_img)  # shape H,W,C dtype=uint8
    # move to device and convert in one step
    t = t_cpu.permute(2, 0, 1).contiguous().unsqueeze(0).to(device=DEVICE, dtype=MODEL_DTYPE)
    t = t / 255.0
    return t

def process(img_rgb: np.ndarray, height) -> np.ndarray:
    """
    Resize BGR/UMat numpy image to target height, keeping aspect ratio.
    """
    if isinstance(img_rgb, cv2.UMat):
        img_rgb = img_rgb.get()
    h0 = img_rgb.shape[0]
    if height < h0:
        width = int(img_rgb.shape[1] / h0 * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return img_rgb

def apply_gamma(depth, gamma=1.2):
    return torch.pow(depth, gamma)

def apply_contrast(depth, factor=1.2):
    mean = depth.mean(dim=(-2, -1), keepdim=True)  # per image mean
    return torch.clamp((depth - mean) * factor + mean, 0, 1)

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


def post_process_depth(depth):
    depth = normalize_tensor(depth).squeeze()
    if is_metric():
        depth = 1.0 - depth
    depth = apply_gamma(depth)
    depth = apply_contrast(depth)
    depth = apply_foreground_scale(depth, scale=FOREGROUND_SCALE)
    depth = anti_alias(depth, strength=AA_STRENGTH)
    depth = normalize_tensor(depth).squeeze()
    return depth
        
# Load Video Depth Anything Model
def get_video_depth_anything_model(model_id=MODEL_ID):
    """ Load Video Depth Anything model from HuggingFace hub. """
    from huggingface_hub import hf_hub_download
    from models.video_depth_anything.vda2_s import VideoDepthAnything
    # Preparation for video depth anything models
    encoder_dict = {'depth-anything/Video-Depth-Anything-Small': 'vits',
                    'depth-anything/Video-Depth-Anything-Base': 'vitb',
                    'depth-anything/Video-Depth-Anything-Large': 'vitl',
                    'depth-anything/Metric-Video-Depth-Anything-Small': 'vits',
                    'depth-anything/Metric-Video-Depth-Anything-Base': 'vitb',
                    'depth-anything/Metric-Video-Depth-Anything-Large': 'vitl'}

    encoder = encoder_dict.get(model_id, 'vits')

    if 'depth-anything/video-depth-anything' in model_id.lower():
        checkpoint_name = f'video_depth_anything_{encoder}.pth'
    elif 'depth-anything/metric-video-depth-anything' in model_id.lower():
        checkpoint_name = f'metric_video_depth_anything_{encoder}.pth'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name, cache_dir=CACHE_PATH)

    model = VideoDepthAnything(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)
    return model.to(DEVICE)

# Load Depth-Anything-V3 Model
def get_da3_model(model_id=MODEL_ID):
    from models.depth_anything_3.api_n import DepthAnything3
    model = DepthAnything3.from_pretrained(model_id, cache_dir=CACHE_PATH)
    return model.to(DEVICE)

# TensorRT Optimization
def optimize_with_tensorrt(onnx_path=ONNX_PATH, trt_path=TRT_PATH, enable_int8=False):
    """
    Convert ONNX model to TensorRT engine using TensorRT's Python API only.
    Supports FP32, FP16, and INT8 precisions based on global flags.
    
    Args:
        onnx_path: Path to the ONNX model file
        trt_path: Path to save the TensorRT engine
        enable_int8: Enable INT8 precision for QDQ models (no calibrator needed)
    
    Returns:
        tuple: (trt_path, fixed_input_shape) where fixed_input_shape is a tuple (N,C,H,W)
               for fixed-dimension models, or None for dynamic models.
        Returns None if compilation fails.
    """
    try:
        if os.path.exists(trt_path) and RECOMPILE_TRT == False:
            print(f"Loaded existing TensorRT engine: {trt_path}")
            # For cached engines, return None for shape - caller should get it from TensorRTEngine
            return trt_path, None
        
        import tensorrt as trt
        
        # Initialize logger and builder
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Load ONNX model
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print("[Error]", parser.get_error(error))
                return None
        
        # Build configuration
        config = builder.create_builder_config()
        
        # Set precision flags based on global configuration
        config.set_flag(trt.BuilderFlag.FP16)
        
        # Enable INT8 for QDQ models - TensorRT reads quantization params from QDQ nodes
        if enable_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("[TensorRT] INT8 mode enabled for QDQ model")
        
        # Set workspace memory (essential for all precision modes) 
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB Workspace 
        
        # Get input tensor info to detect fixed vs dynamic dimensions
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        input_shape_raw = input_tensor.shape  # TensorRT Dims object
        
        # Convert to list to properly handle TensorRT Dims object
        input_shape = []
        for i in range(len(input_shape_raw)):
            dim_value = input_shape_raw[i]
            input_shape.append(dim_value)
        
        # Debug: print the actual shape values
        print(f"[TensorRT Debug] Input name: {input_name}")
        print(f"[TensorRT Debug] Input shape: {input_shape}")
        
        # Check if input has any dynamic dimensions (-1)
        has_dynamic_dims = any(dim == -1 for dim in input_shape)
        
        if has_dynamic_dims:
            # Semi-dynamic or fully dynamic input: create optimization profile
            # For dynamic dims (-1), use range; for fixed dims, use same value in min/opt/max
            profile = builder.create_optimization_profile()
            
            # Build shapes respecting fixed vs dynamic dimensions
            min_shape = []
            opt_shape = []
            max_shape = []
            
            for i, dim in enumerate(input_shape):
                if dim == -1:
                    # Dynamic dimension - use ranges
                    if i == 0:
                        # Batch dimension
                        min_shape.append(1)
                        opt_shape.append(1)
                        max_shape.append(4)  # Reasonable max batch size
                    elif i >= 2:
                        # Spatial dimensions - if user has dynamic spatial dims
                        min_shape.append(224)
                        opt_shape.append((DEPTH_RESOLUTION//14)*14)
                        max_shape.append(3920)
                    else:
                        # Channel dimension (rare to be dynamic)
                        min_shape.append(3)
                        opt_shape.append(3)
                        max_shape.append(3)
                else:
                    # Fixed dimension - must use same value in min/opt/max
                    min_shape.append(dim)
                    opt_shape.append(dim)
                    max_shape.append(dim)
            
            min_shape = tuple(min_shape)
            opt_shape = tuple(opt_shape)
            max_shape = tuple(max_shape)
            
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            print(f"[TensorRT] Dynamic dims detected, profile: min={min_shape}, opt={opt_shape}, max={max_shape}")
            
            # For models with fixed spatial dims, store them
            if len(input_shape) >= 4 and input_shape[2] > 0 and input_shape[3] > 0:
                fixed_input_shape = tuple([1 if d == -1 else d for d in input_shape])
                print(f"[TensorRT] Fixed spatial dimensions: {input_shape[2]}x{input_shape[3]}")
            else:
                fixed_input_shape = None
        else:
            # Fully fixed input: no optimization profile needed
            print(f"[TensorRT] Fully fixed input shape: {input_shape}")
            fixed_input_shape = tuple(input_shape)
        
        # Optional: Enable additional optimizations that work well with FP32 [5](@ref)
        # These optimizations can improve performance regardless of precision
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)  # Enable sparse weights
        config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)  # Reject empty algorithms
        
        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("[Error] TensorRT engine build failed")
            return None
            
        with open(trt_path, "wb") as f:
            f.write(serialized_engine)
        
        print(f"[Main] TensorRT engine saved to {trt_path}")
        return trt_path, fixed_input_shape
        
    except Exception as e:
        print(f"[Error] TensorRT optimization failed: {str(e)}")
        return None
    
# Export to ONNX
def export_to_onnx(model, output_path="depth_model.onnx", device=DEVICE, dtype=DTYPE):
    """
    Export the depth estimation model to ONNX format with dynamic axes.
    """
    dummy_input = torch.randn(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION, device=device, dtype=dtype)
    
    input_names = ["pixel_values"]
    output_names = ["predicted_depth"]
    dynamic_axes = {
        'pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'},
        'predicted_depth': {0: 'batch_size', 1: 'height', 2: 'width'}
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=16,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    print(f"ONNX model generated, TensorRT engine compling may take a while...")

# TensorRT Engine Wrapper Class (Without PyCUDA)
class TensorRTEngine:
    def __init__(self, engine_path, device, dtype):
        """
        Initialize TensorRT engine using binding names instead of deprecated methods.
        """
        self.device = device
        self.dtype = dtype
        self.input_shape = None  # Store input shape for fixed-dimension models
        
        try:
            import tensorrt as trt
            
            # Load TensorRT engine
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            
            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
            
            # Get binding information using TensorRT's tensor mode API
            self.input_binding_indices = []
            self.output_binding_indices = []
            
            for binding in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(binding)
                mode = self.engine.get_tensor_mode(name)
                
                if mode == trt.TensorIOMode.INPUT:
                    self.input_binding_indices.append(binding)
                else:
                    self.output_binding_indices.append(binding)
            
            # Store input name and shape (for fixed-dimension detection)
            if self.input_binding_indices:
                first_input_binding = self.input_binding_indices[0]
                self.input_name = self.engine.get_tensor_name(first_input_binding)
                # Get the input shape from the engine
                engine_input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
                # Check if all dimensions are positive (fixed) or have -1 (dynamic)
                if all(dim > 0 for dim in engine_input_shape):
                    self.input_shape = engine_input_shape
                    print(f"[TensorRT] Engine has fixed input shape: {self.input_shape}")
            else:
                raise RuntimeError("No input bindings found in TensorRT engine")
            
            # Store the actual output name (first output) for dynamic model support
            if self.output_binding_indices:
                first_output_binding = self.output_binding_indices[0]
                self.output_name = self.engine.get_tensor_name(first_output_binding)
            else:
                raise RuntimeError("No output bindings found in TensorRT engine")
            
            # Pre-allocate output tensors
            self.output_shapes = {}
            for binding in self.output_binding_indices:
                name = self.engine.get_tensor_name(binding)
                self.output_shapes[name] = self.engine.get_tensor_shape(name)
            
        except ImportError:
            raise ImportError("TensorRT not available")
    
    def get_fixed_input_size(self):
        """
        Return the fixed input size (H, W) if the engine has fixed dimensions.
        Returns None if the engine has dynamic dimensions.
        """
        if self.input_shape is not None and len(self.input_shape) == 4:
            return (self.input_shape[2], self.input_shape[3])  # (H, W)
        return None

    def __call__(self, tensor):
        """Execute inference with TensorRT using native API."""
        # Set input binding dimensions
        input_shape = tuple(tensor.shape)
        name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(name, input_shape)
        
        # Prepare output tensors
        outputs = {}
        bindings = [None] * self.engine.num_io_tensors

        # Set input binding
        bindings[0] = tensor.data_ptr()
        
        # Allocate output tensors
        for i, binding in enumerate(self.output_binding_indices, 1):
            name = self.engine.get_tensor_name(binding)
            dims = self.context.get_tensor_shape(name)
            shape_tuple = tuple(dims)  # Convert Dims to tuple
            output = torch.empty(shape_tuple, device=self.device, dtype=self.dtype)
            outputs[name] = output
            bindings[binding] = output.data_ptr()
        
        # Execute inference
        self.context.execute_v2(bindings=bindings)
        
        # Return the main output using detected name (supports different models)
        return outputs[self.output_name]


# ONNX Runtime Model Wrapper Class for GPU Inference
# DEPRECATED: This class is no longer used. ONNX files are now compiled directly
# to native TensorRT engines via _load_qdq_tensorrt_engine() for better INT8 support.
# Kept for reference only.
class ONNXModelWrapper:
    """
    DEPRECATED: Use native TensorRT via _load_qdq_tensorrt_engine() instead.
    
    This class used ONNX Runtime with TensorRT Execution Provider, but encountered
    issues with INT8 QDQ models (HasExternalDataInMemory error). Native TensorRT
    compilation provides better compatibility.
    """
    def __init__(self, onnx_path, device_id=0, dtype=torch.float32):
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError("onnxruntime-gpu is required. Install with: pip install onnxruntime-gpu")
        
        self.device_id = device_id
        self.dtype = dtype
        self.device = torch.device(f'cuda:{device_id}')
        
        # Engine cache directory (same folder as ONNX model)
        cache_dir = os.path.dirname(os.path.abspath(onnx_path))
        
        # TensorRT EP options for INT8 QDQ models
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': device_id,
                'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
                'trt_fp16_enable': True,
                'trt_int8_enable': True,  # Critical for INT8 QDQ models
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
            }),
            ('CUDAExecutionProvider', {'device_id': device_id}),  # Fallback
        ]
        
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 2  # WARNING level
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load model from file path (required for TRT engine caching)
        print(f"[ONNX] Loading model: {onnx_path}")
        print(f"[ONNX] TensorRT engine cache: {cache_dir}")
        print(f"[ONNX] First run may take several minutes for TensorRT compilation...")
        
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        
        # Check which provider is active
        active_providers = self.session.get_providers()
        if 'TensorrtExecutionProvider' in active_providers:
            print(f"[ONNX] Running on TensorRT (INT8 enabled)")
        elif 'CUDAExecutionProvider' in active_providers:
            print(f"[ONNX] Running on CUDA (TensorRT unavailable)")
        else:
            raise RuntimeError(f"Failed to load on GPU. Active providers: {active_providers}")
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"[ONNX] Input: {self.input_name}, Shape: {self.input_shape}")
        print(f"[ONNX] Output: {self.output_name}")
        
        # Check for fixed input dimensions
        self.has_fixed_input = all(isinstance(dim, int) for dim in self.input_shape)
        if self.has_fixed_input:
            self.fixed_h = self.input_shape[2]
            self.fixed_w = self.input_shape[3]
            print(f"[ONNX] Fixed input size: {self.fixed_h}x{self.fixed_w}")
        else:
            self.fixed_h = None
            self.fixed_w = None
    
    def __call__(self, tensor):
        """Run inference on GPU via ONNX Runtime."""
        # Convert PyTorch tensor to numpy (ONNX Runtime handles GPU transfer)
        input_np = tensor.cpu().numpy().astype(np.float32)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_np})
        
        # Convert output back to PyTorch tensor on GPU
        return torch.from_numpy(outputs[0]).to(device=self.device, dtype=self.dtype)
    
    def get_fixed_input_size(self):
        """Return fixed input size if model has fixed dimensions."""
        if self.has_fixed_input:
            return (self.fixed_h, self.fixed_w)
        return None
    
    def parameters(self):
        """Compatibility method."""
        return iter([torch.zeros(1)])
    
    def eval(self):
        """Compatibility method."""
        return self


# Model Wrapper Class
class DepthModelWrapper:
    def __init__(self, model_path, device, device_info, dtype, size=None,
                 onnx_path=ONNX_PATH, trt_path=TRT_PATH):
        """
        Wrapper class that handles PyTorch, ONNX, and TensorRT backends.
        """
        self.device = device
        self.device_info = device_info
        self.dtype = dtype
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.trt_path = trt_path
        self.size = size
        self.use_torch_compile = USE_TORCH_COMPILE
        self.onnx_fixed_size = None  # Will store fixed input size for ONNX models
        self.trt_fixed_size = None   # Will store fixed input size for QDQ TensorRT models
        
        # Determine backend based on device
        self.is_cuda = IS_CUDA
        
        # Check if model_path is an ONNX file - use native TensorRT with INT8 support
        if is_onnx_model(model_path):
            if not self.is_cuda:
                raise RuntimeError("ONNX models require CUDA/GPU. Please ensure a CUDA-capable device is available.")
            
            try:
                # Use native TensorRT for QDQ ONNX models (better INT8 support than ONNX Runtime)
                self.backend = "TensorRT"
                self.is_qdq_model = True  # Flag to indicate this is a QDQ ONNX model compiled to TensorRT
                self.model = self._load_qdq_tensorrt_engine(model_path)
                if self.model is None:
                    raise RuntimeError("Failed to compile QDQ ONNX to TensorRT engine")
                print(f"Using backend: {self.backend} (QDQ INT8)")
                return
            except Exception as e:
                print(f"[Error] QDQ TensorRT loading failed: {str(e)}")
                raise
        
        # Standard PyTorch/TensorRT path
        if self.is_cuda and USE_TENSORRT:
            # Use TensorRT backend for CUDA
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            try:
                # First try TensorRT
                self.backend = "TensorRT"
                self.model = self._load_tensorrt_engine()
                if self.model is None:
                    # Fall back to PyTorch if TensorRT fails
                    print("[Error] TensorRT failed, falling back to PyTorch")
                    self.backend = "PyTorch"
                    self.model = self._load_pytorch_model(enable_trt=False)
            except Exception as e:
                print(f"[Error] TensorRT initialization failed: {str(e)}, falling back to PyTorch")
                self.backend = "PyTorch"
                self.model = self._load_pytorch_model(enable_trt=False)
        else:
            # Use PyTorch backend for DirectML/MPS/CPU
            
            # Ignore specific warning message
            warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available")

            self.backend = "PyTorch"
            self.model = self._load_pytorch_model()
        
        print(f"Using backend: {self.backend}")
    
    def _load_pytorch_model(self, enable_trt=USE_TENSORRT):
        """Load the original PyTorch model."""
        # Load model
        if 'video-depth-anything' in MODEL_ID.lower():
            model = get_video_depth_anything_model(MODEL_ID)
        elif 'da3'  in MODEL_ID.lower():
            model = get_da3_model(MODEL_ID)
        else:
            # Load depth model
            model = AutoModelForDepthEstimation.from_pretrained(
                MODEL_ID,
                dtype=torch.float16 if FP16 else torch.float32,
                cache_dir=CACHE_PATH,
                weights_only=True
            ).to(DEVICE)
        
        if FP16 and 'da3' not in MODEL_ID.lower():
            model.half()
        
        if self.is_cuda and 'NVIDIA' in self.device_info and self.use_torch_compile and not enable_trt:
            model = torch.compile(model)
            print("Processing torch.compile with Triton, it may take a while...")
        
        return model.eval()
    
    def _load_tensorrt_engine(self):
        """Load or create TensorRT engine."""
        # First, load PyTorch model to export ONNX
        pytorch_model = self._load_pytorch_model()
        
        # Export to ONNX if not exists
        if RECOMPILE_TRT or not os.path.exists(self.onnx_path):
            export_to_onnx(pytorch_model, self.onnx_path, self.device, self.dtype)
        
        # Build or load TensorRT engine
        result = optimize_with_tensorrt(self.onnx_path, self.trt_path)
        if result is None:
            return None
        
        trt_engine_path, fixed_shape = result
        
        try:
            engine = TensorRTEngine(trt_engine_path, self.device, self.dtype)
            # Get fixed input size from engine (handles both cached and newly compiled)
            fixed_size = engine.get_fixed_input_size()
            if fixed_size is not None:
                self.trt_fixed_size = fixed_size
            return engine
        except Exception as e:
            print(f"[Error] TensorRT engine loading failed: {str(e)}")
            return None
    
    def _load_qdq_tensorrt_engine(self, qdq_onnx_path):
        """
        Load TensorRT engine directly from QDQ ONNX model.
        Skips PyTorch model loading and ONNX export - goes straight to TensorRT compilation.
        
        Args:
            qdq_onnx_path: Path to the QDQ ONNX model file
        
        Returns:
            TensorRTEngine instance or None if compilation fails
        """
        # Generate TRT path based on ONNX filename
        base_name = os.path.splitext(os.path.basename(qdq_onnx_path))[0]
        onnx_dir = os.path.dirname(os.path.abspath(qdq_onnx_path))
        trt_path = os.path.join(onnx_dir, f"{base_name}_int8.trt")
        
        print(f"[TensorRT] Compiling QDQ ONNX to INT8 TensorRT engine...")
        print(f"[TensorRT] Source: {qdq_onnx_path}")
        print(f"[TensorRT] Target: {trt_path}")
        
        # Compile with INT8 enabled for QDQ models
        result = optimize_with_tensorrt(qdq_onnx_path, trt_path, enable_int8=True)
        if result is None:
            print("[Error] Failed to compile QDQ ONNX to TensorRT")
            return None
        
        engine_path, fixed_shape = result
        
        try:
            engine = TensorRTEngine(engine_path, self.device, self.dtype)
            # Get fixed input size from engine (handles both cached and newly compiled)
            fixed_size = engine.get_fixed_input_size()
            if fixed_size is not None:
                self.trt_fixed_size = fixed_size
                print(f"[TensorRT] Model requires fixed input: {self.trt_fixed_size}")
            return engine
        except Exception as e:
            print(f"[Error] TensorRT engine loading failed: {str(e)}")
            return None
    
    def __call__(self, tensor):
        """Run inference using the active backend."""
        # PyTorch and TensorRT backends (ONNX files are now compiled to TensorRT)
        if self.is_cuda:
            with torch.inference_mode():
                with torch.amp.autocast('cuda'):
                    if self.backend == "PyTorch":
                        if "video-depth-anything" in MODEL_ID.lower():
                            return self.model(pixel_values=tensor)
                        elif "da3" in MODEL_ID.lower():
                            return self.model.predict_depth(tensor)
                        return self.model(pixel_values=tensor).predicted_depth
                    else:
                        # TensorRT backend
                        return self.model(tensor)
        else:
            with torch.no_grad():
                if self.backend == "PyTorch":
                    if "video-depth-anything" in MODEL_ID.lower():
                        return self.model(pixel_values=tensor)
                    elif "da3" in MODEL_ID.lower():
                        return self.model.predict_depth(tensor)
                    return self.model(pixel_values=tensor).predicted_depth
                else:
                    return self.model(tensor)

# Initialize model wrapper
model_wraper = DepthModelWrapper(
    model_path=MODEL_ID,
    device=DEVICE,
    device_info=DEVICE_INFO,
    dtype=DTYPE
)

MODEL_DTYPE = next(model_wraper.model.parameters()).dtype if hasattr(model_wraper.model, 'parameters') else DTYPE
if "depthpro" or "zoedepth" or "dpt" in MODEL_ID.lower():
    MEAN = torch.tensor([0.5,0.5,0.5], device=DEVICE).view(1,3,1,1)
    STD = torch.tensor([0.5,0.5,0.5], device=DEVICE).view(1,3,1,1)
else:    
    MEAN = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
    STD = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)
    
if USE_TORCH_COMPILE and IS_CUDA:
    try:
        # Compile the model as before, but SKIP compiling lightweight post-processing functions，avoid FX re-tracing conflicts. These are fast without it.  
        post_process_depth = torch.compile(post_process_depth)
        # Assign to a global or module-level var if needed for access
        globals()['post_process_depth'] = post_process_depth  # Or use a class/module attribute
        
    except Exception as e:
        print(f"[Warning] torch.compile failed: {str(e)}, running without it.")


# Initialize with dummy input for warmup
def warmup_model(model_wraper, steps: int = 3):
    if IS_CUDA:
        with torch.inference_mode():
            with torch.amp.autocast('cuda' , dtype=DTYPE):
                for i in range(steps):
                    dummy = torch.randn(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION,
                                        device=DEVICE, dtype=MODEL_DTYPE)
                    model_wraper(dummy)
    else:
        with torch.no_grad():
            for i in range(steps):
                dummy = torch.randn(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION,
                                    device=DEVICE, dtype=MODEL_DTYPE)
                model_wraper(dummy)
        # print(f"Warmup complete with {steps} iterations.")

warmup_model(model_wraper, steps=3)

lock = Lock()

# Temporal depth stabilizer (EMA)
class DepthStabilizer:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.prev = None
        self.enabled = True
        self.lock = Lock()

    def __call__(self, depth: torch.Tensor):
        if not self.enabled:
            return depth
        with self.lock:
            if self.prev is None or self.prev.shape != depth.shape or self.prev.device != depth.device:
                self.prev = depth.detach().clone()
                return depth
            out = self.alpha * self.prev + (1.0 - self.alpha) * depth
            self.prev = out.detach().clone()
            return out

depth_stabilizer = DepthStabilizer(alpha=0.9)  # increase alpha for more stability
if USE_TORCH_COMPILE and IS_CUDA:
    depth_stabilizer.__call__ = torch.compile(depth_stabilizer.__call__, fullgraph=True)

# Modified predict_depth function with improved TRT and ONNX integration
def predict_depth(image_rgb: np.ndarray, return_tuple=False, use_temporal_smooth: bool = True):
    """
    Returns depth in [0,1], where 1 = near, 0 = far.
    Optionally returns (depth_tensor [H,W], rgb_c [C,H,W]) if return_tuple=True.
    
    Optimized: All resizing and normalization done on GPU for maximum performance.
    Supports PyTorch, TensorRT, and ONNX backends.
    """
    h, w = image_rgb.shape[:2]
    
    # Check if using fixed input size (ONNX or TensorRT with QDQ model)
    is_qdq_model = getattr(model_wraper, 'is_qdq_model', False)
    onnx_fixed_size = getattr(model_wraper, 'onnx_fixed_size', None)
    trt_fixed_size = getattr(model_wraper, 'trt_fixed_size', None)
    
    # Compute target size based on backend
    if is_qdq_model and onnx_fixed_size is not None:
        # QDQ ONNX model with fixed input dimensions - use those
        target_h, target_w = onnx_fixed_size
    elif trt_fixed_size is not None:
        # QDQ TensorRT model with fixed input dimensions
        target_h, target_w = trt_fixed_size
    else:
        # Standard size computation
        scale = DEPTH_RESOLUTION / min(h, w)
        target_h, target_w = int(round(h * scale)), int(round(w * scale))
        
        # Ensure dimensions divisible by 14 for ViT-based models (Depth Anything, etc.)
        if "anything" in MODEL_ID.lower():
            target_h = (target_h // 14) * 14
            target_w = (target_w // 14) * 14
            # Special case: Video-Depth-Anything expects fixed square input
            if "video-depth-anything" in MODEL_ID.lower():
                target_h, target_w = DEPTH_RESOLUTION, DEPTH_RESOLUTION
        else:
            # Fixed square input for other models (e.g., DepthPro, DPT)
            target_h, target_w = DEPTH_RESOLUTION, DEPTH_RESOLUTION

    # EARLY GPU TRANSFER + FULL GPU PREPROCESSING
    # Convert NumPy -> Torch tensor and move to device early
    tensor = torch.from_numpy(image_rgb).to(device=DEVICE, dtype=torch.float32, non_blocking=True)
    
    if return_tuple:
        # Keep original RGB for return (CHW format)
        rgb_c = tensor.permute(2, 0, 1).contiguous()  # [C, H, W]
        tensor = rgb_c.unsqueeze(0) / 255.0  # [1, C, H, W]
    else:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # [1, C, H, W]

    # Resize on GPU using bilinear interpolation (very fast on CUDA)
    if (h, w) != (target_h, target_w):
        tensor = F.interpolate(
            tensor,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )

    # Normalize using ImageNet stats (or custom) — on GPU
    tensor = (tensor - MEAN) / STD
    
    # For QDQ TensorRT models (from ONNX), keep float32; for others, use MODEL_DTYPE
    if is_qdq_model:
        tensor = tensor.to(dtype=torch.float32).contiguous()
    else:
        tensor = tensor.to(dtype=MODEL_DTYPE).contiguous()

    # MODEL INFERENCE
    if is_qdq_model:
        # QDQ TensorRT model - direct call with inference mode
        with torch.inference_mode():
            depth = model_wraper(tensor)
    elif "video-depth-anything" in MODEL_ID.lower():
        with torch.no_grad():
            depth = model_wraper(tensor)
    else:
        if IS_CUDA:
            with torch.inference_mode():
                with torch.amp.autocast('cuda', dtype=DTYPE):
                    depth = model_wraper(tensor)
        else:
            with torch.no_grad():
                depth = model_wraper(tensor)

    # POST-PROCESSING (already GPU-based and optionally compiled)
    with torch.no_grad():
        depth = post_process_depth(depth)  # includes gamma, contrast, foreground scale, AA, etc.

    # Optional temporal stabilization (EMA)
    if use_temporal_smooth:
        depth = depth_stabilizer(depth)

    # Resize depth back to original input resolution (on GPU)
    depth = F.interpolate(
        depth.unsqueeze(0).unsqueeze(0),  # [1, 1, h_model, w_model]
        size=(h, w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)  # [H, W]

    # Return
    if return_tuple:
        return depth, rgb_c  # rgb_c is [C, H_orig, W_orig], uint8 range on GPU
    else:
        return depth  # [H_orig, W_orig], float32 in [0,1] on GPU

# Global cache (module-level)
_FONT_CACHE = {}

def build_font(device="cpu", dtype=torch.float32):
    """
    Build (once) and cache FPS font tensors per (device, dtype).
    Safe for CUDA / DirectML / MPS / CPU.
    """
    key = (str(device), dtype)

    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    # Build once
    chars = sorted(font_dict.keys())

    font_tensor = torch.stack([
        torch.tensor(
            [[1.0 if c == "1" else 0.0 for c in row] for row in font_dict[ch]],
            device=device,
            dtype=dtype,
        )
        for ch in chars
    ])  # [num_chars, 5, 3]

    _FONT_CACHE[key] = (chars, font_tensor)
    return chars, font_tensor

# module-level cache
_FPS_MASK_CACHE = {
    "mask": None,
    "frame": 0,
    "interval": 10,  # update every N frames
}

def overlay_fps(rgb: torch.Tensor, fps: float):
    device, dtype = rgb.device, rgb.dtype
    H, W = rgb.shape[1:]

    cache = _FPS_MASK_CACHE
    cache["frame"] += 1

    # Rebuild only every N frames
    if cache["mask"] is None or cache["frame"] % cache["interval"] == 0:
        chars, font_tensor = build_font(device, dtype)
        txt = f"FPS: {fps:.1f}"

        idxs = torch.tensor(
            [chars.index(ch) if ch in chars else chars.index(" ") for ch in txt],
            device=device
        )

        scale = max(1, min(8, H // 60))
        char_h, char_w = 5 * scale, 3 * scale
        spacing = scale
        margin_x, margin_y = 2 * scale, 2 * scale

        glyphs = font_tensor[idxs]
        glyphs = glyphs.repeat_interleave(scale, 1).repeat_interleave(scale, 2)

        mask = torch.zeros((H, W), device=device, dtype=dtype)

        for i, glyph in enumerate(glyphs):
            x0 = margin_x + i * (char_w + spacing)
            y0 = margin_y
            x1 = min(W, x0 + char_w)
            y1 = min(H, y0 + char_h)
            if x0 < W and y0 < H:
                mask[y0:y1, x0:x1] = torch.maximum(
                    mask[y0:y1, x0:x1],
                    glyph[:y1 - y0, :x1 - x0]
                )

        cache["mask"] = mask

    alpha = cache["mask"].unsqueeze(0)
    color = torch.tensor([0.0, 255.0, 0.0], device=device, dtype=dtype).view(3,1,1)
    return rgb * (1 - alpha) + color * alpha


def _coerce_rgb_tensor(rgb_c, depth: torch.Tensor) -> torch.Tensor:
    """Convert input RGB to CHW tensor on the same device/dtype as depth."""
    if isinstance(rgb_c, np.ndarray):
        rgb = torch.from_numpy(rgb_c).to(device=depth.device, dtype=depth.dtype)
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            rgb = rgb.permute(2, 0, 1)
    else:
        rgb = rgb_c.to(device=depth.device, dtype=depth.dtype)
    return rgb


def _pad_to_aspect_tensor(tensor, target_ratio=(16, 9)):
    _, h, w = tensor.shape
    t_w, t_h = target_ratio
    r_img, r_t = w / h, t_w / t_h
    if abs(r_img - r_t) < 1e-3:
        return tensor
    if r_img > r_t:  # too wide -> pad height
        new_h = int(round(w / r_t))
        pad_top = (new_h - h) // 2
        return F.pad(tensor, (0, 0, pad_top, new_h - h - pad_top))
    new_w = int(round(h * r_t))
    pad_left = (new_w - w) // 2
    return F.pad(tensor, (pad_left, new_w - w - pad_left, 0, 0))


def _generate_stereo_pair_core(rgb: torch.Tensor,
                               depth: torch.Tensor,
                               ipd_uv=0.064,
                               depth_ratio=1.0,
                               fill_16_9=FILL_16_9,
                               device=DEVICE):
    """Return left/right eye tensors in 0-255 range."""
    # Cast to float32 for DirectML compatibility (avoids float64 ops)
    if IS_DIRECTML:
        rgb = rgb.to(dtype=torch.float32, device=device)
        depth = depth.to(dtype=torch.float32, device=device)
        
    C, H, W = rgb.shape
    img = rgb.unsqueeze(0)  # [1,C,H,W]
    
    depth_strength = 0.05
    inv = 1.0 - depth * depth_ratio
    max_px = ipd_uv * W
    shifts = inv * max_px * depth_strength
    
    # CUDA fast path: grid_sample
    if not IS_DIRECTML:
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=DTYPE).view(1, 1, W).expand(1, H, W)
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=DTYPE).view(1, H, 1).expand(1, H, W)
        shift_norm = shifts * (2.0 / (W - 1))
        grid_left = torch.stack([xs + shift_norm, ys], dim=-1)
        grid_right = torch.stack([xs - shift_norm, ys], dim=-1)
        if IS_MPS:
            grid_left, grid_right = grid_left.clamp(-1,1), grid_right.clamp(-1,1)
            left = F.grid_sample(img, grid_left, mode="bilinear",
                                padding_mode="zeros", align_corners=False)[0]
            right = F.grid_sample(img, grid_right, mode="bilinear",
                                padding_mode="zeros", align_corners=False)[0]
        else:
            left = F.grid_sample(img, grid_left, mode="bilinear",
                                padding_mode="border", align_corners=False)[0]
            right = F.grid_sample(img, grid_right, mode="bilinear",
                                padding_mode="border", align_corners=False)[0]
    # Fallback path: vectorized gather (DirectML / MPS / CPU safe)
    else:
        base = torch.arange(W, device=device, dtype=torch.int64).view(1, -1).expand(H, -1)
        # Ensure shifts is float32 for addition
        shifts = shifts.to(dtype=torch.float32)
        coords_left = (base.to(dtype=torch.float32) + shifts).clamp(0, W - 1).long()  # [H,W]
        coords_right = (base.to(dtype=torch.float32) - shifts).clamp(0, W - 1).long()  # [H,W]
        # Left eye
        gather_idx_left = coords_left.unsqueeze(0).expand(C, H, W).unsqueeze(0)  # [1,C,H,W]
        left = torch.gather(img.expand(1, C, H, W), 3, gather_idx_left)[0]  # [C,H,W]
        # Right eye
        gather_idx_right = coords_right.unsqueeze(0).expand(C, H, W).unsqueeze(0)
        right = torch.gather(img.expand(1, C, H, W), 3, gather_idx_right)[0]

    if fill_16_9:
        left = _pad_to_aspect_tensor(left)
        right = _pad_to_aspect_tensor(right)
    return left.clamp(0, 255), right.clamp(0, 255)


def _arrange_stereo_output(left: torch.Tensor, right: torch.Tensor, display_mode="Half-SBS") -> torch.Tensor:
    if display_mode == "TAB":
        out = torch.cat([left, right], dim=1)
    else:
        out = torch.cat([left, right], dim=2)
    if display_mode != "Full-SBS":
        out = F.interpolate(out.unsqueeze(0), size=left.shape[1:], mode="area")[0]
    return out.clamp(0, 255)


def _tensor_to_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    return tensor.to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()


def depth_to_image(depth) -> np.ndarray:
    """Convert normalized depth to an 8-bit grayscale image."""
    if hasattr(depth, "detach"):
        return depth.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    depth_arr = np.asarray(depth, dtype=np.float32)
    return np.clip(np.rint(depth_arr * 255.0), 0, 255).astype(np.uint8)


def make_sbs_core(rgb: torch.Tensor,
                  depth: torch.Tensor,
                  ipd_uv=0.064,
                  depth_ratio=1.0,
                  display_mode="Half-SBS",
                  fill_16_9=FILL_16_9,
                  device=DEVICE) -> torch.Tensor:
    """
    Core tensor operations for side-by-side stereo.
    Keeps CUDA fast path (grid_sample) and fallback path (gather).
    Compatible with torch.compile.
    Inputs:
        rgb: [C,H,W] float tensor
        depth: [H,W] float tensor
    Returns:
        SBS image [C,H,W] float tensor (0-255 range)
    """
    left, right = _generate_stereo_pair_core(rgb, depth, ipd_uv, depth_ratio, fill_16_9, device)
    return _arrange_stereo_output(left, right, display_mode)

def make_sbs(rgb_c, depth, ipd_uv=0.064, depth_ratio=1.0, display_mode="Half-SBS", fps=None):
    """
    Full function: adds optional FPS overlay and converts output to numpy uint8.
    Calls `make_sbs_core` for tensor computations (torch.compile compatible).
    """
    if depth.dim() == 3 and depth.shape[0] == 1:
        depth = depth[0]
    rgb = _coerce_rgb_tensor(rgb_c, depth)

    # Optional FPS overlay can stay in Python side (avoids torch.compile recompiles)
    if fps is not None:
        rgb = overlay_fps(rgb, fps)  # your existing overlay function

    sbs_tensor = make_sbs_core(rgb, depth, ipd_uv, depth_ratio, display_mode)
    return _tensor_to_uint8_hwc(sbs_tensor)


def make_stereo_views(rgb_c, depth, ipd_uv=0.064, depth_ratio=1.0, display_mode="Half-SBS", fps=None):
    """Return left, right, and SBS images as uint8 numpy arrays."""
    if depth.dim() == 3 and depth.shape[0] == 1:
        depth = depth[0]
    rgb = _coerce_rgb_tensor(rgb_c, depth)
    if fps is not None:
        rgb = overlay_fps(rgb, fps)
    left, right = _generate_stereo_pair_core(rgb, depth, ipd_uv, depth_ratio)
    sbs = _arrange_stereo_output(left, right, display_mode)
    return _tensor_to_uint8_hwc(left), _tensor_to_uint8_hwc(right), _tensor_to_uint8_hwc(sbs)


def _save_image_array(path: str, image: np.ndarray):
    image = np.ascontiguousarray(image)
    if image.ndim == 2:
        ok = cv2.imwrite(path, image)
    else:
        ok = cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Failed to save image: {path}")


def save_image_outputs(input_path: str,
                       depth,
                       rgb_c,
                       ipd_uv=0.064,
                       depth_ratio=1.0,
                       display_mode="Half-SBS") -> dict:
    """Save depth, left, right, and SBS images next to the input image."""
    base_path, _ = os.path.splitext(input_path)
    depth_img = depth_to_image(depth)
    left_img, right_img, sbs_img = make_stereo_views(
        rgb_c,
        depth,
        ipd_uv=ipd_uv,
        depth_ratio=depth_ratio,
        display_mode=display_mode,
    )
    output_paths = {
        "depth": f"{base_path}_depth.png",
        "left": f"{base_path}_left.png",
        "right": f"{base_path}_right.png",
        "sbs": f"{base_path}_sbs.png",
    }
    _save_image_array(output_paths["depth"], depth_img)
    _save_image_array(output_paths["left"], left_img)
    _save_image_array(output_paths["right"], right_img)
    _save_image_array(output_paths["sbs"], sbs_img)
    return output_paths

if USE_TORCH_COMPILE and IS_CUDA:
    make_sbs_core = torch.compile(make_sbs_core)