import depthai as dai

with dai.Device() as dev:
    try:
        cams = dev.getConnectedCameras()
        print('getConnectedCameras ->', cams)
    except Exception as e:
        print('getConnectedCameras not available:', e)
    try:
        info = dev.getDeviceInfo()
        print('getDeviceInfo ->', info)
    except Exception as e:
        print('getDeviceInfo not available:', e)
    try:
        mx = dev.getMxId()
        print('MXID ->', mx)
    except Exception as e:
        print('getMxId not available:', e)
