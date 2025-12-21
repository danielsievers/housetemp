import unittest
import os
import ast

class TestArchitecture(unittest.TestCase):
    """
    Enforce architectural boundaries.
    Core Logic (housetemp/housetemp) must NOT import from HASS Integration (housetemp/*.py or ..).
    """
    
    def test_core_lib_isolation(self):
        """Scan core lib files for forbidden imports."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        core_lib_dir = os.path.join(base_dir, "custom_components", "housetemp", "housetemp")
        
        forbidden_patterns = [
            "from ..const", 
            "from ..", 
            "custom_components.housetemp.const"
        ]
        
        for filename in os.listdir(core_lib_dir):
            if not filename.endswith(".py"):
                continue
                
            path = os.path.join(core_lib_dir, filename)
            with open(path, "r") as f:
                content = f.read()
                
            # AST parsing is robust, but string search catches comments.
            # We want code imports.
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                    level = node.level
                    
                    # Check relative imports: level 2 means ".."
                    if level and level >= 2:
                        self.fail(f"Forbidden Parent Import in {filename}: from ..{module or ''} import ...")
                    
                    # Check absolute imports referencing HASS layer
                    if module and "custom_components.housetemp" in module:
                        # Allow importing self (core lib)
                        if "custom_components.housetemp.housetemp" in module:
                            continue
                        
                        # Forbidden: importing from parent specific configs
                        if module.endswith("const") or module == "custom_components.housetemp":
                            self.fail(f"Forbidden Upgrade Dependency in {filename}: import {module}")

if __name__ == '__main__':
    unittest.main()
