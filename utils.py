def print_log(*content, file=None, file_path=None):
    print(*content)
    if file is not None:
        print(*content, file=file)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*content, file=f)


def sec_to_time(secconds):
    sec = int(secconds)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f'{h:d}:{m:02d}:{s:02d}'


def list_devices():
    from tensorflow.python.client import device_lib
    gpu_devices = device_lib.list_local_devices()
    print('Num. of GPU devices:', len(gpu_devices))
    for gpu in gpu_devices:
        print('-------')
        print('Name:', gpu.name)
        print('Type:', gpu.device_type)
        print('Memory limit:', int(gpu.memory_limit / 1024 / 1024), 'MB')
    print('=========')
