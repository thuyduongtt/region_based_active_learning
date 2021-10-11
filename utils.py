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
