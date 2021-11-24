# type: ignore

import os
import shutil


def pytest_sessionstart(session):  # pylint: disable=unused-argument
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    shutil.rmtree('./tests/output', ignore_errors=True)
    os.makedirs('./tests/output', exist_ok=True)


# def pytest_sessionfinish(session, exitstatus):
#     """
#     Called after whole test run finished, right before
#     returning the exit status to the system.
#     """


# def pytest_unconfigure(config):
#     """
#     called before test process is exited.
#     """
