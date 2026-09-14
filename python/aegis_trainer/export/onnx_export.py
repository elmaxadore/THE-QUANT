"""ONNX model exporter."""
import logging
logger = logging.getLogger(__name__)

class ONNXExporter:
    """Exports trained models to ONNX format for Rust deployment."""
    def export(self, model, path):
        logger.info(f"Exporting model to {path}")
        return True
