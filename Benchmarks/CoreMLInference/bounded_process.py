"""Run one benchmark child with a timeout and process-group cleanup."""
import os
import signal
import subprocess


def execute(command, log, timeout):
    with log.open('w') as f:
        process = subprocess.Popen([str(x) for x in command], stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        finally:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
        if code:
            raise RuntimeError(f'Exit {code}: {log}')
