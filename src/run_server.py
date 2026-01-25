#!/usr/bin/env python3
"""
Run the Maqam Detector API server.
"""
import sys
import os

# Add src directory to path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

# Now import and run
if __name__ == "__main__":
    import uvicorn
    
    # Import the app after path is set
    import importlib.util
    spec = importlib.util.spec_from_file_location("api", os.path.join(src_dir, "api.py"))
    api_module = importlib.util.module_from_spec(spec)
    
    # Fix relative imports by pre-importing the modules
    from SignalProcessor import SignalProcessor
    from TonicFinder import TonicFinder
    from SequenceNormalizer import SequenceNormalizer
    from MLPClassifier import MLPClassifier
    from JinsLibrary import JinsAnalyzer, MAQAM_STRUCTURE
    from MarkovSeyirClassifier import MarkovSeyirClassifier
    
    # Now we can import MaqamBrain and MaqamTrainer
    # But we need to fix their imports first, so let's do inline monkey patching
    import types
    
    # Create the module namespaces
    sys.modules['MLPClassifier'] = types.ModuleType('MLPClassifier')
    sys.modules['MLPClassifier'].MLPClassifier = MLPClassifier
    sys.modules['JinsLibrary'] = types.ModuleType('JinsLibrary')
    sys.modules['JinsLibrary'].JinsAnalyzer = JinsAnalyzer
    sys.modules['JinsLibrary'].MAQAM_STRUCTURE = MAQAM_STRUCTURE
    sys.modules['MarkovSeyirClassifier'] = types.ModuleType('MarkovSeyirClassifier')
    sys.modules['MarkovSeyirClassifier'].MarkovSeyirClassifier = MarkovSeyirClassifier
    
    print("Starting Maqam Detector API on http://localhost:8000")
    print("Open http://localhost:8000/app/ in your browser")
    
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
