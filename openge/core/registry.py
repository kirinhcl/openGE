"""Global registry for extensible component management."""

from typing import Dict, Type, Callable, Any, Optional


class Registry:
    """Registry for managing and creating model components dynamically."""
    
    def __init__(self, name: str = "Registry"):
        """
        Initialize registry.
        
        Args:
            name: Registry name
        """
        self.name = name
        self._registry: Dict[str, Type] = {}
    
    def register(self, name: str = None):
        """
        Decorator for registering a component.
        
        Args:
            name: Component name (defaults to class name)
            
        Returns:
            Decorator function
        """
        def decorator(cls: Type) -> Type:
            component_name = name or cls.__name__
            if component_name in self._registry:
                raise ValueError(f"Component '{component_name}' already registered in {self.name}")
            self._registry[component_name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> Type:
        """
        Get registered component by name.
        
        Args:
            name: Component name
            
        Returns:
            Registered component class
            
        Raises:
            KeyError: If component not found
        """
        if name not in self._registry:
            raise KeyError(f"Component '{name}' not found in {self.name}. "
                          f"Available: {list(self._registry.keys())}")
        return self._registry[name]
    
    def build(self, name: str, **kwargs) -> Any:
        """
        Build (instantiate) a registered component.
        
        Args:
            name: Component name
            **kwargs: Arguments to pass to component constructor
            
        Returns:
            Instantiated component
        """
        component_cls = self.get(name)
        return component_cls(**kwargs)
    
    def list(self) -> list:
        """List all registered component names."""
        return list(self._registry.keys())
    
    def is_registered(self, name: str) -> bool:
        """Check if component is registered."""
        return name in self._registry


# Global registries for different component types
encoder_registry = Registry("EncoderRegistry")
fusion_registry = Registry("FusionRegistry")
model_registry = Registry("ModelRegistry")
head_registry = Registry("HeadRegistry")
interpretability_registry = Registry("InterpretabilityRegistry")

__all__ = [
    "Registry",
    "encoder_registry",
    "fusion_registry",
    "model_registry",
    "head_registry",
    "interpretability_registry",
]
