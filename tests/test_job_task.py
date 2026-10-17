import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BASH = shutil.which('bash')


def seconds_left(value):
    result = subprocess.run([BASH, '-c', 'set -u; source hpc/time_left.sh; seconds_left "$1"', '_', value],
                            cwd=REPO, capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_seconds_left_reads_the_formats_of_squeue():
    assert seconds_left('1-02:03:04') == '93784'
    assert seconds_left('2:03:04') == '7384'
    assert seconds_left('08:09') == '489'      # leading zeros are not octal
    assert seconds_left('45') == '45'
    for value in ('UNLIMITED', 'NOT_SET', 'INVALID', ''):
        assert seconds_left(value) == ''        # no time limit known: the job gets no stop of its own
