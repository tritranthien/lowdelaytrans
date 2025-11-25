import pyaudiowpatch as pyaudio

def list_all_devices():
    """List all available audio devices (input and loopback)"""
    p = pyaudio.PyAudio()
    
    devices = {
        'input': [],
        'loopback': []
    }
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        
        # Check if it's an input device
        if info['maxInputChannels'] > 0:
            device_info = {
                'index': i,
                'name': info['name'],
                'sample_rate': int(info['defaultSampleRate']),
                'channels': info['maxInputChannels']
            }
            
            # Categorize as loopback or regular input
            if 'loopback' in info['name'].lower():
                devices['loopback'].append(device_info)
            else:
                devices['input'].append(device_info)
    
    p.terminate()
    return devices

def get_device_by_name(device_name):
    """Get device index by name"""
    p = pyaudio.PyAudio()
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['name'] == device_name:
            p.terminate()
            return i
    
    p.terminate()
    return None

def get_default_output_device():
    """Get default output device for loopback"""
    p = pyaudio.PyAudio()
    
    try:
        # Get WASAPI info
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_device_index = wasapi_info["defaultOutputDevice"]
        default_device = p.get_device_info_by_index(default_device_index)
        
        device_name = default_device['name']
        
        # Find matching loopback device
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if device_name in info['name'] and 'loopback' in info['name'].lower():
                p.terminate()
                return {
                    'index': i,
                    'name': info['name'],
                    'sample_rate': int(info['defaultSampleRate'])
                }
        
        p.terminate()
        return None
        
    except Exception as e:
        p.terminate()
        raise e
