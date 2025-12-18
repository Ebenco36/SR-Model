#!/usr/bin/env python3
"""
ROBUST HARDWARE-AGNOSTIC MODEL COMPARISON
Optimized for Mac MPS, NVIDIA GPU, or CPU

Features:
- ✅ Automatic hardware detection and optimization
- ✅ Mixed precision training for all devices
- ✅ Memory optimization and gradient checkpointing
- ✅ Hardware-specific batch size tuning
- ✅ Progress tracking with rich visualizations
- ✅ Comprehensive error handling and recovery
- ✅ Model checkpointing and resumption
"""

import os
import sys
import json
import torch
import numpy as np
import platform
import psutil
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Rich logging for better visualization
try:
    from rich.logging import RichHandler
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Install rich for better visualizations: pip install rich")

# ============================================================
# HARDWARE DETECTION & OPTIMIZATION
# ============================================================

class HardwareOptimizer:
    """Optimize training for available hardware"""
    
    @staticmethod
    def detect_hardware():
        """Detect available hardware and return optimal settings"""
        hardware_info = {
            'system': platform.system(),
            'processor': platform.processor(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'available_ram_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        # Check for MPS (Apple Silicon)
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Check for CUDA
        cuda_available = torch.cuda.is_available()
        
        # Check for multiple GPUs
        if cuda_available:
            hardware_info['cuda_device_count'] = torch.cuda.device_count()
            hardware_info['cuda_devices'] = []
            for i in range(torch.cuda.device_count()):
                hardware_info['cuda_devices'].append({
                    'name': torch.cuda.get_device_name(i),
                    'memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
                })
        
        # Determine best device
        if mps_available:
            device = torch.device("mps")
            device_type = "MPS (Apple Silicon)"
            hardware_info['device'] = device
            hardware_info['device_type'] = device_type
            # Apply MPS optimizations
            HardwareOptimizer.optimize_mps()
            
        elif cuda_available:
            device = torch.device("cuda:0")
            device_type = f"CUDA ({torch.cuda.get_device_name(0)})"
            hardware_info['device'] = device
            hardware_info['device_type'] = device_type
            # CUDA-specific optimizations
            torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
            torch.cuda.set_device(0)
            # Set memory fraction for CUDA
            try:
                torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of memory
            except AttributeError:
                pass  # Some versions might not have this
            
        else:
            device = torch.device("cpu")
            device_type = "CPU"
            hardware_info['device'] = device
            hardware_info['device_type'] = device_type
        
        hardware_info['device_str'] = str(device)
        hardware_info['mps_available'] = mps_available
        hardware_info['cuda_available'] = cuda_available
        
        return hardware_info
    
    @staticmethod
    def optimize_mps():
        """Apply MPS-specific optimizations"""
        if platform.system() == 'Darwin':
            # Set MPS memory management - use a safe value
            try:
                # Clear any existing problematic values first
                if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
                    current_val = os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
                    try:
                        if float(current_val) > 1.0:
                            os.environ.pop('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
                    except ValueError:
                        os.environ.pop('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
                
                # Set to a safe value
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.75'  # Safe 75% of memory
                
                # Enable fallback to CPU for unsupported operations
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                
                print("MPS optimizations applied")
            except Exception as e:
                print(f"Warning: MPS optimization failed: {e}")
    
    @staticmethod
    def get_optimized_batch_size(model_size_mb: float, device_info: Dict) -> Dict:
        """Calculate optimal batch sizes based on hardware and model size"""
        batch_sizes = {
            'train': 8,
            'eval': 16,
            'grad_accumulation': 1
        }
        
        if device_info['device_type'].startswith('CUDA'):
            vram_gb = device_info['cuda_devices'][0]['memory_gb']
            
            if vram_gb >= 24:  # A100, 4090, etc.
                batch_sizes = {'train': 32, 'eval': 64, 'grad_accumulation': 1}
            elif vram_gb >= 16:  # V100, 3080, etc.
                batch_sizes = {'train': 16, 'eval': 32, 'grad_accumulation': 1}
            elif vram_gb >= 8:  # 2080, 3070, etc.
                batch_sizes = {'train': 8, 'eval': 16, 'grad_accumulation': 2}
            else:  # Low VRAM cards
                batch_sizes = {'train': 4, 'eval': 8, 'grad_accumulation': 4}
                
        elif device_info['device_type'].startswith('MPS'):
            # M1/M2 with unified memory
            ram_gb = device_info['ram_gb']
            
            if ram_gb >= 32:  # M1 Max/Ultra, M2 Max
                batch_sizes = {'train': 16, 'eval': 32, 'grad_accumulation': 1}
            elif ram_gb >= 16:  # M1 Pro, M2 Pro
                batch_sizes = {'train': 8, 'eval': 16, 'grad_accumulation': 2}
            else:  # Base M1/M2
                batch_sizes = {'train': 4, 'eval': 8, 'grad_accumulation': 4}
                
        else:  # CPU
            ram_gb = device_info['ram_gb']
            cpu_count = device_info['cpu_count']
            
            if ram_gb >= 32 and cpu_count >= 16:
                batch_sizes = {'train': 4, 'eval': 8, 'grad_accumulation': 8}
            elif ram_gb >= 16 and cpu_count >= 8:
                batch_sizes = {'train': 2, 'eval': 4, 'grad_accumulation': 16}
            else:
                batch_sizes = {'train': 1, 'eval': 2, 'grad_accumulation': 32}
        
        # Adjust for model size (larger models need smaller batches)
        if model_size_mb > 1000:  # >1GB model
            batch_sizes['train'] = max(1, batch_sizes['train'] // 2)
            batch_sizes['eval'] = max(1, batch_sizes['eval'] // 2)
            batch_sizes['grad_accumulation'] = min(64, batch_sizes['grad_accumulation'] * 2)
        
        return batch_sizes
    
    @staticmethod
    def optimize_memory():
        """Optimize PyTorch memory usage"""
        # Enable memory-efficient attention if available
        if hasattr(torch.backends, 'mha'):
            torch.backends.mha.set_fastpath_enabled(True)
        
        # Enable deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        # Enable TF32 on Ampere GPUs for faster training
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Clear cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    @staticmethod
    def get_device_capabilities(device_info: Dict) -> Dict:
        """Get device capabilities and limitations"""
        capabilities = {
            'mixed_precision': False,
            'gradient_checkpointing': False,
            'distributed_training': False,
            'max_sequence_length': 512
        }
        
        if device_info['device_type'].startswith('CUDA'):
            capabilities['mixed_precision'] = True
            capabilities['gradient_checkpointing'] = True
            capabilities['distributed_training'] = device_info['cuda_device_count'] > 1
            
            # Check compute capability for sequence length limits
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:  # Volta or newer
                capabilities['max_sequence_length'] = 2048
            else:
                capabilities['max_sequence_length'] = 512
                
        elif device_info['device_type'].startswith('MPS'):
            capabilities['mixed_precision'] = True  # MPS supports mixed precision
            capabilities['gradient_checkpointing'] = False  # Not well supported
            capabilities['max_sequence_length'] = 1024  # MPS memory limits
            
        else:  # CPU
            capabilities['mixed_precision'] = False
            capabilities['gradient_checkpointing'] = True
            capabilities['max_sequence_length'] = 512  # CPU memory limits
        
        return capabilities

# ============================================================
# CONFIGURATION MANAGEMENT
# ============================================================

@dataclass
class TrainingConfig:
    """Training configuration optimized for hardware"""
    model_id: str
    model_name: str
    output_dir: Path
    hardware_info: Dict = field(default_factory=dict)
    capabilities: Dict = field(default_factory=dict)
    
    # Training parameters
    max_length: int = 384
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Hardware optimized parameters (will be set automatically)
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False
    tf32: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    
    def __post_init__(self):
        """Initialize hardware-optimized parameters"""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set hardware-specific parameters
        if self.hardware_info.get('device_type', '').startswith('CUDA'):
            self.fp16 = True
            self.tf32 = True
            self.dataloader_num_workers = min(4, os.cpu_count() or 1)
            self.dataloader_pin_memory = True
            
        elif self.hardware_info.get('device_type', '').startswith('MPS'):
            self.bf16 = True  # MPS prefers bfloat16
            self.dataloader_num_workers = 0  # MPS doesn't support multiprocessing well
            self.dataloader_pin_memory = False
            
        else:  # CPU
            self.dataloader_num_workers = min(8, os.cpu_count() or 1)
            self.dataloader_pin_memory = False
    
    def to_training_args(self, model_size_mb: float = 500) -> Dict:
        """Convert to HuggingFace TrainingArguments format"""
        # Adjust batch sizes based on model size
        batch_config = HardwareOptimizer.get_optimized_batch_size(
            model_size_mb, self.hardware_info
        )
        
        self.train_batch_size = batch_config['train']
        self.eval_batch_size = batch_config['eval']
        self.gradient_accumulation_steps = batch_config['grad_accumulation']
        
        return {
            'output_dir': str(self.output_dir),
            'overwrite_output_dir': True,
            'num_train_epochs': self.epochs,
            'per_device_train_batch_size': self.train_batch_size,
            'per_device_eval_batch_size': self.eval_batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_ratio': self.warmup_ratio,
            'logging_dir': str(self.output_dir / "logs"),
            'logging_strategy': "steps",
            'logging_steps': self.logging_steps,
            'evaluation_strategy': "steps",
            'eval_steps': self.eval_steps,
            'save_strategy': "steps",
            'save_steps': self.save_steps,
            'save_total_limit': self.save_total_limit,
            'load_best_model_at_end': True,
            'metric_for_best_model': "eval_loss",
            'greater_is_better': False,
            'fp16': self.fp16,
            'bf16': self.bf16,
            'tf32': self.tf32,
            'gradient_checkpointing': self.gradient_checkpointing,
            'dataloader_num_workers': self.dataloader_num_workers,
            'dataloader_pin_memory': self.dataloader_pin_memory,
            'remove_unused_columns': True,
            'label_names': ["start_positions", "end_positions"],
            'report_to': "none",  # Disable wandb/tensorboard unless configured
            'ddp_find_unused_parameters': False,
            'optim': "adamw_torch",
            'lr_scheduler_type': "linear",
            'seed': 42,
        }

# ============================================================
# ENHANCED LOGGING WITH PROGRESS TRACKING
# ============================================================

class EnhancedLogger:
    """Enhanced logging with progress tracking and hardware monitoring"""
    
    def __init__(self, log_dir: Path, use_rich: bool = RICH_AVAILABLE):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_rich = use_rich
        
        # Setup console
        if self.use_rich:
            self.console = Console()
            self.progress = None
        else:
            self.console = None
        
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Hardware monitor
        self.hardware_info = HardwareOptimizer.detect_hardware()
        self._log_hardware_info()
    
    def _log_hardware_info(self):
        """Log detailed hardware information"""
        self.logger.info("=" * 80)
        self.logger.info("HARDWARE DETECTION & OPTIMIZATION")
        self.logger.info("=" * 80)
        self.logger.info(f"System: {self.hardware_info['system']} {self.hardware_info['platform']}")
        self.logger.info(f"Processor: {self.hardware_info['processor']}")
        self.logger.info(f"CPU Cores: {self.hardware_info['cpu_count']}")
        self.logger.info(f"RAM: {self.hardware_info['ram_gb']:.1f} GB")
        self.logger.info(f"Available RAM: {self.hardware_info['available_ram_gb']:.1f} GB")
        
        if self.hardware_info['cuda_available']:
            self.logger.info(f"CUDA Devices: {self.hardware_info['cuda_device_count']}")
            for i, device in enumerate(self.hardware_info['cuda_devices']):
                self.logger.info(f"  GPU {i}: {device['name']} ({device['memory_gb']:.1f} GB)")
        
        self.logger.info(f"Selected Device: {self.hardware_info['device_type']}")
        self.logger.info("=" * 80)
    
    def create_progress_bar(self, total: int, description: str = "Processing"):
        """Create a rich progress bar"""
        if self.use_rich and self.console:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=True
            )
            self.task_id = self.progress.add_task(description, total=total)
            self.progress.start()
        else:
            self.progress = None
            self.logger.info(f"{description}: 0/{total}")
    
    def update_progress(self, advance: int = 1):
        """Update progress bar"""
        if self.progress:
            self.progress.update(self.task_id, advance=advance)
        elif not self.use_rich:
            # Simple console update
            sys.stdout.write('.')
            sys.stdout.flush()
    
    def complete_progress(self):
        """Complete and clear progress bar"""
        if self.progress:
            self.progress.stop()
            self.progress = None
        elif not self.use_rich:
            sys.stdout.write('\n')
            sys.stdout.flush()
    
    def log_memory_usage(self):
        """Log current memory usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            self.logger.debug(f"GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
        
        process = psutil.Process()
        memory_used = process.memory_info().rss / (1024**3)
        self.logger.debug(f"RAM Usage: {memory_used:.2f} GB")
    
    def log_table(self, title: str, data: List[Dict], columns: List[str]):
        """Log data as a table"""
        if self.use_rich and self.console and data:
            table = Table(title=title)
            for col in columns:
                table.add_column(col)
            for row in data:
                table.add_row(*[str(row.get(col, "")) for col in columns])
            self.console.print(table)
        elif data:
            self.logger.info(title)
            for row in data:
                self.logger.info(" | ".join([f"{col}: {row.get(col, '')}" for col in columns]))

# ============================================================
# CALLBACKS
# ============================================================

class EarlyStoppingCallback:
    """Custom early stopping callback"""
    
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.001):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.patience_counter = 0

# ============================================================
# ROBUST MODEL MANAGER
# ============================================================

class RobustModelManager:
    """Manage model loading, training, and evaluation with hardware optimization"""
    
    def __init__(self, config: TrainingConfig, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
        self.device = self.config.hardware_info['device']
        self.capabilities = self.config.capabilities
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model with hardware optimizations"""
        try:
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering
            
            self.logger.logger.info(f"Loading model: {self.config.model_id}")
            
            # Handle specific models with different tokenizer requirements
            try:
                if 'deberta-v3' in self.config.model_id.lower():
                    # DeBERTa-v3 needs special handling
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
                    except Exception as e:
                        if "tiktoken" in str(e):
                            # Try without fast tokenizer
                            self.logger.logger.warning("DeBERTa-v3: Trying without fast tokenizer...")
                            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, use_fast=False)
                        else:
                            raise
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            except Exception as e:
                self.logger.logger.error(f"Failed to load tokenizer: {e}")
                return False
            
            # Configure model loading for hardware
            model_kwargs = {
                'low_cpu_mem_usage': True,
            }
            
            # Enable gradient checkpointing for memory efficiency
            if self.config.gradient_checkpointing:
                model_kwargs['use_cache'] = False
            
            # Load model
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.config.model_id,
                **model_kwargs
            )
            
            # Apply hardware-specific optimizations
            self._optimize_model_for_hardware()
            
            # Move to device with error handling
            try:
                self.model = self.model.to(self.device)
                self.logger.logger.info(f"Model loaded successfully on {self.device}")
            except RuntimeError as e:
                if "invalid low watermark ratio" in str(e):
                    self.logger.logger.error(f"MPS memory error: {e}")
                    self.logger.logger.error("Trying with CPU instead...")
                    self.device = torch.device("cpu")
                    self.config.hardware_info['device_type'] = "CPU (fallback)"
                    self.model = self.model.to(self.device)
                    self.logger.logger.info(f"Model loaded on CPU fallback")
                else:
                    raise
            
            # Calculate model size
            model_size = self._get_model_size_mb()
            self.logger.logger.info(f"Model size: {model_size:.1f} MB")
            
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Failed to load model: {e}")
            import traceback
            self.logger.logger.error(traceback.format_exc())
            return False
    
    def _optimize_model_for_hardware(self):
        """Apply hardware-specific optimizations to model"""
        
        # Gradient checkpointing
        if self.config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            self.logger.logger.info("Enabled gradient checkpointing")
        
        # Mixed precision preparation
        if self.config.fp16 or self.config.bf16:
            from torch.cuda.amp import autocast
            self.autocast = autocast
        
        # For MPS, try channels last memory format if supported
        if str(self.device) == 'mps' and hasattr(self.model, 'to'):
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                self.logger.logger.info("Enabled channels last memory format for MPS")
            except Exception as e:
                self.logger.logger.debug(f"Could not set channels last: {e}")
    
    def _get_model_size_mb(self) -> float:
        """Calculate model size in megabytes"""
        if self.model is None:
            return 0
        
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024**2)
        return size_mb
    
    def train(self, train_dataset, eval_dataset, num_train_examples: int):
        """Train model with hardware optimization"""
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            from transformers.trainer_utils import get_last_checkpoint
            
            # Prepare training arguments
            model_size_mb = self._get_model_size_mb()
            training_args_dict = self.config.to_training_args(model_size_mb)
            
            # Log training configuration
            self.logger.logger.info("Training Configuration:")
            for key, value in training_args_dict.items():
                if 'batch_size' in key or 'lr' in key or 'steps' in key:
                    self.logger.logger.info(f"  {key}: {value}")
            
            # Check for existing checkpoint
            last_checkpoint = None
            if os.path.isdir(self.config.output_dir):
                last_checkpoint = get_last_checkpoint(self.config.output_dir)
                if last_checkpoint is not None:
                    self.logger.logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            
            # Initialize Trainer
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(**training_args_dict),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                ),
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.config.early_stopping_patience,
                        early_stopping_threshold=self.config.early_stopping_threshold
                    )
                ]
            )
            
            # Train
            self.logger.logger.info("Starting training...")
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
            
            # Save final model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Save training metrics
            metrics = train_result.metrics
            metrics['train_samples'] = num_train_examples
            metrics['model_size_mb'] = model_size_mb
            metrics['hardware'] = self.config.hardware_info['device_type']
            metrics['training_time'] = str(timedelta(seconds=int(metrics.get('train_runtime', 0))))
            
            metrics_path = self.config.output_dir / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            self.logger.logger.info(f"Training completed in {metrics.get('train_runtime', 0):.0f} seconds")
            self.logger.logger.info(f"Final training loss: {metrics.get('train_loss', 'N/A'):.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.logger.error(f"Training failed: {e}")
            import traceback
            self.logger.logger.error(traceback.format_exc())
            return None
    
    def evaluate(self, test_dataset, max_samples: int = 1000):
        """Evaluate model on test dataset"""
        try:
            sample_count = min(len(test_dataset), max_samples)
            self.logger.logger.info(f"Evaluating model on {sample_count} samples...")
            
            # Simple evaluation without pipeline for compatibility
            exact_matches = []
            f1_scores = []
            
            self.logger.create_progress_bar(sample_count, "Evaluating")
            
            for i, example in enumerate(test_dataset[:sample_count]):
                try:
                    # Tokenize the input
                    inputs = self.tokenizer(
                        example['question'],
                        example['context'],
                        truncation=True,
                        padding='max_length',
                        max_length=self.config.max_length,
                        return_tensors='pt'
                    )
                    
                    # Move inputs to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get model prediction
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Get start and end positions
                    start_idx = torch.argmax(outputs.start_logits)
                    end_idx = torch.argmax(outputs.end_logits)
                    
                    # Convert to answer text
                    input_ids = inputs['input_ids'][0]
                    predicted_answer = self.tokenizer.decode(input_ids[start_idx:end_idx+1])
                    
                    true_answers = example.get('answers', {}).get('text', [])
                    if not true_answers:
                        true_answers = [example.get('answer', '')]
                    
                    # Calculate exact match
                    exact_match = any(predicted_answer.strip() == ta.strip() for ta in true_answers)
                    exact_matches.append(exact_match)
                    
                    # Calculate F1 score
                    if predicted_answer and true_answers:
                        best_f1 = max(self._calculate_f1(predicted_answer, ta) for ta in true_answers)
                        f1_scores.append(best_f1)
                    else:
                        f1_scores.append(0.0)
                    
                except Exception as e:
                    self.logger.logger.debug(f"Error evaluating sample {i}: {e}")
                    exact_matches.append(False)
                    f1_scores.append(0.0)
                
                self.logger.update_progress()
            
            self.logger.complete_progress()
            
            # Calculate metrics
            if exact_matches:
                metrics = {
                    'exact_match': float(np.mean(exact_matches)),
                    'f1_score': float(np.mean(f1_scores)),
                    'num_evaluated': len(exact_matches),
                    'evaluation_hardware': str(self.device)
                }
            else:
                metrics = {
                    'exact_match': 0.0,
                    'f1_score': 0.0,
                    'num_evaluated': 0,
                    'evaluation_hardware': str(self.device)
                }
            
            # Save evaluation metrics
            eval_path = self.config.output_dir / "evaluation_metrics.json"
            with open(eval_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.logger.info(f"Evaluation complete:")
            self.logger.logger.info(f"  Exact Match: {metrics['exact_match']:.2%}")
            self.logger.logger.info(f"  F1 Score: {metrics['f1_score']:.2%}")
            
            return metrics
            
        except Exception as e:
            self.logger.logger.error(f"Evaluation failed: {e}")
            import traceback
            self.logger.logger.error(traceback.format_exc())
            return None
    
    def _calculate_f1(self, predicted: str, true: str) -> float:
        """Calculate F1 score between predicted and true answer"""
        if not predicted or not true:
            return 0.0
        
        # Tokenize
        pred_tokens = set(predicted.lower().split())
        true_tokens = set(true.lower().split())
        
        if not pred_tokens or not true_tokens:
            return 0.0
        
        # Calculate intersection
        common = pred_tokens.intersection(true_tokens)
        
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(true_tokens) if true_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)

# ============================================================
# MAIN COMPARISON SYSTEM
# ============================================================

class HardwareAgnosticModelComparator:
    """Main comparison system with hardware optimization"""
    
    def __init__(self, qa_dataset_dir: Path, output_dir: Path):
        self.dataset_dir = Path(qa_dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = EnhancedLogger(self.output_dir / "logs")
        
        # Detect hardware
        self.hardware_info = HardwareOptimizer.detect_hardware()
        self.capabilities = HardwareOptimizer.get_device_capabilities(self.hardware_info)
        
        # Hardware optimization
        HardwareOptimizer.optimize_memory()
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Results storage
        self.results = {}
        self.comparison_table = []
        
    def _load_dataset(self):
        """Load and preprocess dataset"""
        from datasets import Dataset, DatasetDict
        import json
        
        self.logger.logger.info("Loading QA dataset...")
        
        dataset_dict = {}
        
        for split in ['train', 'validation', 'test']:
            split_path = self.dataset_dir / split / f"{split}.jsonl"
            
            if not split_path.exists():
                self.logger.logger.warning(f"Split {split} not found at {split_path}")
                continue
                
            examples = []
            with open(split_path, 'r', encoding='utf-8') as f:
                for line in f:
                    examples.append(json.loads(line.strip()))
            
            dataset_dict[split] = Dataset.from_list(examples)
            self.logger.logger.info(f"Loaded {len(examples)} examples from {split}")
        
        return DatasetDict(dataset_dict)
    
    def compare_models(self, model_list: List[Dict]):
        """Compare multiple models"""
        
        self.logger.logger.info("=" * 80)
        self.logger.logger.info("MODEL COMPARISON STARTED")
        self.logger.logger.info("=" * 80)
        
        for model_info in model_list:
            self.logger.logger.info(f"\n{'='*40}")
            self.logger.logger.info(f"PROCESSING: {model_info['name']}")
            self.logger.logger.info(f"{'='*40}")
            
            try:
                # Create model output directory
                model_output_dir = self.output_dir / model_info['name'].replace("/", "_").replace(" ", "_")
                
                # Create training config
                config = TrainingConfig(
                    model_id=model_info['model_id'],
                    model_name=model_info['name'],
                    output_dir=model_output_dir,
                    hardware_info=self.hardware_info,
                    capabilities=self.capabilities,
                    max_length=self.capabilities['max_sequence_length'],
                    gradient_checkpointing=self.capabilities['gradient_checkpointing'],
                    epochs=model_info.get('epochs', 3)
                )
                
                # Initialize model manager
                model_manager = RobustModelManager(config, self.logger)
                
                # Load model
                if not model_manager.load_model():
                    self.logger.logger.error(f"Failed to load model {model_info['name']}")
                    continue
                
                # Train model (skip if no training data)
                train_metrics = None
                if 'train' in self.dataset and len(self.dataset['train']) > 0:
                    if 'validation' in self.dataset and len(self.dataset['validation']) > 0:
                        train_metrics = model_manager.train(
                            self.dataset['train'],
                            self.dataset['validation'],
                            len(self.dataset['train'])
                        )
                    else:
                        self.logger.logger.warning("No validation data available, skipping training")
                else:
                    self.logger.logger.warning("No training data available")
                
                # Evaluate model
                eval_metrics = None
                if 'test' in self.dataset and len(self.dataset['test']) > 0:
                    eval_metrics = model_manager.evaluate(self.dataset['test'])
                else:
                    self.logger.logger.warning("No test data available")
                
                # Store results
                model_result = {
                    'name': model_info['name'],
                    'model_id': model_info['model_id'],
                    'hardware': self.config.hardware_info['device_type'],
                    'train_metrics': train_metrics,
                    'eval_metrics': eval_metrics,
                    'parameters_millions': model_manager._get_model_size_mb() / 1024,  # Convert to millions
                    'config': config.__dict__,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results[model_info['name']] = model_result
                
                # Add to comparison table
                em_percent = f"{eval_metrics.get('exact_match', 0)*100:.1f}" if eval_metrics else 'N/A'
                f1_percent = f"{eval_metrics.get('f1_score', 0)*100:.1f}" if eval_metrics else 'N/A'
                train_loss = f"{train_metrics.get('train_loss', 'N/A'):.4f}" if train_metrics else 'N/A'
                train_time = f"{train_metrics.get('train_runtime', 'N/A'):.0f}" if train_metrics else 'N/A'
                param_m = f"{model_result['parameters_millions']:.1f}" if model_result['parameters_millions'] > 0 else 'N/A'
                
                self.comparison_table.append({
                    'Model': model_info['name'],
                    'Hardware': self.hardware_info['device_type'],
                    'EM (%)': em_percent,
                    'F1 (%)': f1_percent,
                    'Train Loss': train_loss,
                    'Time (s)': train_time,
                    'Size (M)': param_m
                })
                
                # Save intermediate results
                self._save_results()
                
                # Clear memory
                del model_manager
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.logger.error(f"Error processing {model_info['name']}: {e}")
                import traceback
                self.logger.logger.error(traceback.format_exc())
                continue
        
        # Generate final report
        self._generate_final_report()
        
        self.logger.logger.info("\n" + "=" * 80)
        self.logger.logger.info("COMPARISON COMPLETE")
        self.logger.logger.info("=" * 80)
        
        return self.results
    
    def _save_results(self):
        """Save intermediate results"""
        results_path = self.output_dir / "comparison_results.json"
        with open(results_path, 'w') as f:
            # Convert Path objects to strings
            import copy
            results_copy = copy.deepcopy(self.results)
            for key, value in results_copy.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, Path):
                            results_copy[key][subkey] = str(subvalue)
            json.dump(results_copy, f, indent=2, default=str)
    
    def _generate_final_report(self):
        """Generate comprehensive comparison report"""
        from datetime import datetime
        
        report = {
            'summary': {
                'total_models': len(self.results),
                'hardware': self.hardware_info,
                'capabilities': self.capabilities,
                'timestamp': datetime.now().isoformat(),
                'dataset_stats': {
                    split: len(self.dataset[split]) for split in self.dataset
                }
            },
            'detailed_results': self.results,
            'comparison_table': self.comparison_table,
        }
        
        # Only generate recommendations if we have results
        if self.results:
            report['recommendations'] = self._generate_recommendations()
        else:
            report['recommendations'] = {}
        
        # Save JSON report
        report_path = self.output_dir / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        # Log summary table
        if self.comparison_table:
            self.logger.log_table("Model Comparison Results", self.comparison_table, 
                                 ['Model', 'Hardware', 'EM (%)', 'F1 (%)', 'Train Loss', 'Time (s)', 'Size (M)'])
    
    def _generate_recommendations(self):
        """Generate recommendations based on results"""
        if not self.results:
            return {}
        
        # Filter models with evaluation metrics
        valid_results = {k: v for k, v in self.results.items() 
                        if v.get('eval_metrics') and v['eval_metrics'].get('f1_score', 0) > 0}
        
        if not valid_results:
            return {}
        
        # Find best by different criteria
        try:
            best_by_f1 = max(valid_results.items(), 
                            key=lambda x: x[1].get('eval_metrics', {}).get('f1_score', 0))
        except ValueError:
            best_by_f1 = (None, {})
        
        try:
            best_by_speed = min(valid_results.items(),
                              key=lambda x: x[1].get('train_metrics', {}).get('train_runtime', float('inf')))
        except ValueError:
            best_by_speed = (None, {})
        
        # Find most efficient (F1 per parameter)
        efficiency_scores = {}
        for name, result in valid_results.items():
            f1 = result.get('eval_metrics', {}).get('f1_score', 0)
            params = result.get('parameters_millions', 1)
            efficiency_scores[name] = f1 / params if params > 0 else 0
        
        try:
            best_by_efficiency = max(efficiency_scores.items(), key=lambda x: x[1]) if efficiency_scores else (None, 0)
        except ValueError:
            best_by_efficiency = (None, 0)
        
        recommendations = {}
        
        if best_by_f1[0]:
            recommendations['best_overall'] = {
                'model': best_by_f1[0],
                'f1_score': best_by_f1[1].get('eval_metrics', {}).get('f1_score', 0),
                'reason': 'Highest F1 score'
            }
        
        if best_by_speed[0] and best_by_speed[1].get('train_metrics', {}).get('train_runtime'):
            recommendations['fastest_training'] = {
                'model': best_by_speed[0],
                'training_time': best_by_speed[1].get('train_metrics', {}).get('train_runtime', 0),
                'reason': 'Fastest training time'
            }
        
        if best_by_efficiency[0]:
            recommendations['most_efficient'] = {
                'model': best_by_efficiency[0],
                'efficiency_score': best_by_efficiency[1],
                'reason': 'Best F1 per parameter'
            }
        
        recommendations['hardware_specific'] = {
            'recommendation': self._get_hardware_specific_recommendation()
        }
        
        return recommendations
    
    def _get_hardware_specific_recommendation(self) -> str:
        """Get hardware-specific recommendations"""
        device_type = self.hardware_info['device_type']
        
        if device_type.startswith('CUDA'):
            vram = self.hardware_info.get('cuda_devices', [{}])[0].get('memory_gb', 0)
            if vram >= 16:
                return "Use large batch sizes (16-32) and enable mixed precision for best performance"
            elif vram >= 8:
                return "Use moderate batch sizes (8-16) with gradient accumulation"
            else:
                return "Use small batch sizes (4-8) with heavy gradient accumulation"
        
        elif device_type.startswith('MPS'):
            ram = self.hardware_info.get('ram_gb', 0)
            if ram >= 32:
                return "M1/M2 Max/Ultra: Use batch size 8-16 with bfloat16"
            elif ram >= 16:
                return "M1/M2 Pro: Use batch size 4-8 with gradient accumulation"
            else:
                return "Base M1/M2: Use batch size 2-4 with heavy gradient accumulation"
        
        else:  # CPU
            cpu_count = self.hardware_info.get('cpu_count', 1)
            ram = self.hardware_info.get('ram_gb', 0)
            
            if cpu_count >= 8 and ram >= 16:
                return "High-end CPU: Use batch size 2-4 with 8 workers"
            elif cpu_count >= 4 and ram >= 8:
                return "Mid-range CPU: Use batch size 1-2 with 4 workers"
            else:
                return "Low-end CPU: Use batch size 1 with 2 workers"
    
    def _generate_markdown_report(self, report: Dict):
        """Generate markdown report"""
        md_path = self.output_dir / "comparison_report.md"
        
        with open(md_path, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Hardware Info
            f.write("## Hardware Information\n\n")
            hw = report['summary']['hardware']
            f.write(f"- **System**: {hw['system']} {hw['platform']}\n")
            f.write(f"- **Processor**: {hw['processor']}\n")
            f.write(f"- **CPU Cores**: {hw['cpu_count']}\n")
            f.write(f"- **RAM**: {hw['ram_gb']:.1f} GB\n")
            f.write(f"- **Device**: {hw['device_type']}\n\n")
            
            # Dataset Info
            f.write("## Dataset Information\n\n")
            for split, count in report['summary']['dataset_stats'].items():
                f.write(f"- **{split.capitalize()}**: {count} examples\n")
            f.write("\n")
            
            # Results Table
            if report['comparison_table']:
                f.write("## Results Summary\n\n")
                f.write("| Model | Hardware | EM (%) | F1 (%) | Train Loss | Time (s) | Size (M) |\n")
                f.write("|-------|----------|--------|--------|------------|----------|----------|\n")
                
                for row in report['comparison_table']:
                    f.write(f"| {row['Model']} | {row['Hardware']} | {row['EM (%)']} | {row['F1 (%)']} | {row['Train Loss']} | {row['Time (s)']} | {row['Size (M)']} |\n")
                f.write("\n")
            
            # Recommendations
            if report['recommendations']:
                recs = report['recommendations']
                
                f.write("## Recommendations\n\n")
                
                if 'best_overall' in recs and recs['best_overall']:
                    f.write(f"### Best Overall Model\n")
                    f.write(f"- **Model**: {recs['best_overall']['model']}\n")
                    f.write(f"- **F1 Score**: {recs['best_overall']['f1_score']:.2%}\n")
                    f.write(f"- **Reason**: {recs['best_overall']['reason']}\n\n")
                
                if 'fastest_training' in recs and recs['fastest_training']:
                    f.write(f"### Fastest Training\n")
                    f.write(f"- **Model**: {recs['fastest_training']['model']}\n")
                    f.write(f"- **Training Time**: {recs['fastest_training']['training_time']:.0f} seconds\n")
                    f.write(f"- **Reason**: {recs['fastest_training']['reason']}\n\n")
                
                if 'most_efficient' in recs and recs['most_efficient']:
                    f.write(f"### Most Efficient (F1 per Parameter)\n")
                    f.write(f"- **Model**: {recs['most_efficient']['model']}\n")
                    f.write(f"- **Efficiency Score**: {recs['most_efficient']['efficiency_score']:.4f}\n")
                    f.write(f"- **Reason**: {recs['most_efficient']['reason']}\n\n")
                
                if 'hardware_specific' in recs and recs['hardware_specific']:
                    f.write(f"### Hardware-Specific Tips\n")
                    f.write(f"{recs['hardware_specific']['recommendation']}\n\n")
            
            # Detailed Results
            if report['detailed_results']:
                f.write("## Detailed Results\n\n")
                for model_name, result in report['detailed_results'].items():
                    f.write(f"### {model_name}\n\n")
                    f.write(f"- **Model ID**: {result['model_id']}\n")
                    f.write(f"- **Parameters**: {result.get('parameters_millions', 'N/A'):.1f}M\n")
                    f.write(f"- **Hardware**: {result['hardware']}\n")
                    
                    if result.get('eval_metrics'):
                        f.write(f"- **Exact Match**: {result['eval_metrics'].get('exact_match', 0):.2%}\n")
                        f.write(f"- **F1 Score**: {result['eval_metrics'].get('f1_score', 0):.2%}\n")
                    
                    if result.get('train_metrics'):
                        f.write(f"- **Training Time**: {result['train_metrics'].get('train_runtime', 0):.0f} seconds\n")
                        f.write(f"- **Training Loss**: {result['train_metrics'].get('train_loss', 'N/A'):.4f}\n")
                    
                    f.write("\n")

# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Robust Hardware-Agnostic Model Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full comparison on all hardware
  python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/
  
  # Use specific models only
  python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/ \\
    --models bert scibert pubmedbert
  
  # Quick test with minimal training
  python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/ \\
    --epochs 1 --test-samples 100
  
  # Force CPU usage (skip GPU/MPS)
  python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/ --cpu-only
        """
    )
    
    parser.add_argument("--qa-dataset", required=True,
                       help="Path to QA dataset directory")
    parser.add_argument("--output", default="./model_comparison_results",
                       help="Output directory for results")
    parser.add_argument("--models", nargs="+",
                       choices=['bert', 'scibert', 'pubmedbert', 'biobert', 'deberta'],
                       help="Models to compare (default: all)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,
                       help="Override automatic batch size detection")
    parser.add_argument("--test-samples", type=int, default=1000,
                       help="Number of samples for evaluation")
    parser.add_argument("--cpu-only", action="store_true",
                       help="Force CPU usage (skip GPU/MPS)")
    parser.add_argument("--no-train", action="store_true",
                       help="Skip training, only evaluate")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Define models to compare - FIXED MODEL IDs
    all_models = [
        {
            'name': 'BERT-base-cased',
            'model_id': 'bert-base-cased',
            'description': 'Standard BERT model with case sensitivity'
        },
        {
            'name': 'SciBERT',
            'model_id': 'allenai/scibert_scivocab_cased',
            'description': 'BERT trained on scientific literature'
        },
        {
            'name': 'PubMedBERT',
            'model_id': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
            'description': 'BERT trained on PubMed abstracts'
        },
        {
            'name': 'BioBERT',
            'model_id': 'dmis-lab/biobert-base-cased-v1.1',
            'description': 'BERT trained on biomedical literature'
        },
        {
            'name': 'DeBERTa-v3',
            'model_id': 'microsoft/deberta-v3-base',
            'description': 'Improved BERT architecture with disentangled attention'
        }
    ]
    
    # Filter models if specified
    if args.models:
        model_mapping = {
            'bert': 'BERT-base-cased',
            'scibert': 'SciBERT',
            'pubmedbert': 'PubMedBERT',
            'biobert': 'BioBERT',
            'deberta': 'DeBERTa-v3'
        }
        selected_names = [model_mapping[m] for m in args.models]
        models_to_compare = [m for m in all_models if m['name'] in selected_names]
    else:
        models_to_compare = all_models
    
    print("\n" + "="*80)
    print("ROBUST HARDWARE-AGNOSTIC MODEL COMPARISON")
    print("="*80)
    print(f"Models to compare: {len(models_to_compare)}")
    print(f"Output directory: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Test samples: {args.test_samples}")
    print("="*80 + "\n")
    
    # Initialize comparator
    comparator = HardwareAgnosticModelComparator(
        qa_dataset_dir=Path(args.qa_dataset),
        output_dir=Path(args.output)
    )
    
    # Update epochs in models
    for model in models_to_compare:
        model['epochs'] = args.epochs
    
    # Run comparison
    results = comparator.compare_models(models_to_compare)
    
    # Print final summary
    print("\n" + "="*80)
    print("COMPARISON COMPLETE - SUMMARY")
    print("="*80)
    
    if comparator.comparison_table:
        try:
            from tabulate import tabulate
            print(tabulate(comparator.comparison_table, headers="keys", tablefmt="grid"))
        except ImportError:
            # Simple table if tabulate not available
            print("Model\tHardware\tEM (%)\tF1 (%)\tTrain Loss\tTime (s)\tSize (M)")
            for row in comparator.comparison_table:
                print(f"{row['Model']}\t{row['Hardware']}\t{row['EM (%)']}\t{row['F1 (%)']}\t{row['Train Loss']}\t{row['Time (s)']}\t{row['Size (M)']}")
    
    # Best model recommendation
    if comparator.results:
        # Filter models with evaluation metrics
        valid_results = {k: v for k, v in comparator.results.items() 
                        if v.get('eval_metrics') and v['eval_metrics'].get('f1_score', 0) > 0}
        
        if valid_results:
            best_model = max(valid_results.items(), 
                            key=lambda x: x[1].get('eval_metrics', {}).get('f1_score', 0))
            print(f"\n✅ **BEST MODEL**: {best_model[0]}")
            print(f"   F1 Score: {best_model[1].get('eval_metrics', {}).get('f1_score', 0):.2%}")
            model_dir_name = best_model[0].replace("/", "_").replace(" ", "_")
            print(f"   Saved to: {args.output}/{model_dir_name}/")
        else:
            print("\n⚠️  No models were successfully evaluated")
    
    print("\n📊 Full reports available:")
    print(f"   JSON Report: {args.output}/comparison_report.json")
    print(f"   Markdown Report: {args.output}/comparison_report.md")
    print(f"   Logs: {args.output}/logs/training.log")
    print("="*80)

# ============================================================
# QUICK START SCRIPT
# ============================================================

def quick_start():
    """Quick start guide and setup"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║      Robust Model Comparison - Quick Start Guide             ║
╚══════════════════════════════════════════════════════════════╝

1. INSTALLATION:
   pip install torch transformers datasets rich psutil tabulate

2. PREPARE DATA:
   python qa_generator.py --ner ner_results.jsonl --output qa_dataset/

3. RUN COMPARISON:
   python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/

4. HARDWARE SUPPORT:
   - Apple Silicon (M1/M2/M3): Uses MPS backend automatically
   - NVIDIA GPU: Uses CUDA with mixed precision
   - CPU: Optimized for multi-core processing

5. OPTIONS:
   --models bert scibert       # Compare specific models
   --epochs 5                  # Train for 5 epochs
   --batch-size 16             # Override automatic batch size
   --test-samples 500          # Evaluate on 500 samples
   --cpu-only                  # Force CPU usage

6. OUTPUT:
   - results/comparison_report.md    # Human-readable report
   - results/comparison_report.json  # Detailed JSON results
   - results/MODEL_NAME/             # Trained model checkpoints
   - results/logs/                   # Training logs

7. TIPS:
   - For Apple Silicon: Ensure PyTorch 2.0+ with MPS support
   - For GPU training: Use --batch-size auto (default)
   - For large datasets: Reduce --test-samples for faster evaluation
   - For debugging: Add --verbose flag

Example workflow:
  1. python qa_generator.py --ner ner_results.jsonl --output qa_dataset/
  2. python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/
  3. Check results/comparison_report.md for best model selection
    """)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Check for help or quick start
    import sys
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        quick_start()
        sys.exit(0)
    
    # Run main comparison
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Comparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# # Basic run (auto-detects hardware)
# python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/

# # Force CPU (skip GPU/MPS)
# python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/ --cpu-only

# # Compare specific models
# python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/ \
#   --models bert pubmedbert biobert

# # Custom training
# python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/ \
#   --epochs 5 --test-samples 500 --batch-size 16
  
  
  
# # MPS is auto-detected, uses bfloat16
# python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/

# # Monitor MPS memory
# export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8  # Use 80% of memory


# # CUDA auto-detected, uses mixed precision
# python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/

# # Multi-GPU support (if available)
# CUDA_VISIBLE_DEVICES=0,1 python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/



# # Optimized CPU training with multiprocessing
# python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/ --cpu-only

# # Set number of workers
# OMP_NUM_THREADS=8 python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/ --cpu-only




# #!/usr/bin/env python3
# """
# USE THE BEST MODEL FROM COMPARISON
# """

# import json
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# # Load comparison results
# with open('results/comparison_report.json', 'r') as f:
#     report = json.load(f)

# # Get best model
# best_model = report['recommendations']['best_overall']['model']
# print(f"Best model: {best_model}")

# # Load the model
# model_path = Path("results") / best_model.replace("/", "_")
# tokenizer = AutoTokenizer.from_pretrained(str(model_path))
# model = AutoModelForQuestionAnswering.from_pretrained(str(model_path))

# # Create QA pipeline
# qa_pipeline = pipeline(
#     "question-answering",
#     model=model,
#     tokenizer=tokenizer,
#     device=0 if torch.cuda.is_available() else -1
# )

# # Use it
# context = "A systematic review of 25 studies from PubMed..."
# question = "How many studies were included?"
# result = qa_pipeline(question=question, context=context)
# print(f"Answer: {result['answer']}")





# # If training is interrupted, it auto-resumes from last checkpoint
# python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/

# # Force fresh start
# rm -rf results/
# python -m src.SR.model_run --qa-dataset qa_dataset/ --output results/



# # View logs in real-time
# tail -f results/logs/training.log

# # Monitor GPU memory (Linux/Mac)
# watch -n 1 nvidia-smi  # NVIDIA
# watch -n 1 vmmap <pid>  # Apple Silicon

# # Monitor CPU usage
# htop  # or top