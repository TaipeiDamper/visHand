from config.settings import Settings
from core.detector import Point3D
from core.interpreter import GestureInterpreter

def test():
    s = Settings()
    interp = GestureInterpreter(s)
    
    # Create fake landmarks to avoid mediapipe initialization
    fake_lms = [Point3D(0.5, 0.5, 0.5)] * 21
    
    # Try processing
    payload = interp.process(fake_lms, "RIGHT", 12345.0, 1)
    
    print("Success! Payload Intent:", payload["state"]["intent"])

if __name__ == "__main__":
    test()
