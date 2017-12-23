from subprocess import call


def run_path_no_inf(path):
    call(['python', 'run.py', path,
          '--inference', '0',
          '--rechunk', '0'])


if __name__ == '__main__':
    paths = ['',
             '']
    for path in paths:
        run_path_no_inf(path)
