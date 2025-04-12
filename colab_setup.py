"""
Helper module for setting up Colab environment for control vector experiments.
"""
import os
import sys
import subprocess
import importlib
from typing import List, Optional


def is_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed."""
    return importlib.util.find_spec(package_name) is not None


def install_requirements(requirements_file: str = 'requirements.txt', 
                         essential_packages: Optional[List[str]] = None) -> None:
    """Install required packages from requirements file."""
    
    # Install essential packages first if specified
    if essential_packages:
        missing_essentials = [pkg for pkg in essential_packages if not is_package_installed(pkg)]
        if missing_essentials:
            print(f"Installing essential packages: {', '.join(missing_essentials)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing_essentials)
    
    # Install from requirements file
    if os.path.exists(requirements_file):
        print(f"Installing packages from {requirements_file}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", requirements_file])
        print("✓ All packages installed successfully!")
    else:
        print(f"Warning: {requirements_file} not found. Skipping package installation.")


def setup_environment() -> None:
    """Set up the complete environment for working with control vectors."""
    
    # Essential packages for working with control vectors
    essential_packages = [
        "torch", 
        "transformers", 
        "numpy", 
        "pandas", 
        "matplotlib"
    ]
    
    # Install packages
    install_requirements(essential_packages=essential_packages)
    
    # Set CUDA visible devices if needed
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Add project root to Python path
    if "/content/repeng" not in sys.path:
        sys.path.append("/content/repeng")
    
    # Check for GPU
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    except ImportError:
        print("PyTorch not installed. GPU status unknown.")
    
    print("Environment setup complete!")


def save_to_drive(source_path: str, drive_path: str) -> None:
    """
    Save file or directory to Google Drive.
    
    Args:
        source_path: Path to the file/directory to save
        drive_path: Destination path in Google Drive
    """
    import subprocess
    import os
    
    if not os.path.exists(source_path):
        print(f"Error: Source path {source_path} doesn't exist.")
        return
    
    drive_dir = os.path.dirname(drive_path)
    if not os.path.exists(drive_dir):
        os.makedirs(drive_dir)
    
    if os.path.isdir(source_path):
        # Copy directory
        subprocess.run(["cp", "-r", source_path, drive_path], check=True)
    else:
        # Copy file
        subprocess.run(["cp", source_path, drive_path], check=True)
    
    print(f"✓ Saved to Drive: {drive_path}")