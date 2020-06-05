import os
import pytest

import westac.common.stagger_wrapper as stagger

def test_create_wrapper_by_env():

    stagger_home = os.path.abspath('./lib/stagger')

    os.environ['STAGGER_HOME'] = stagger_home

    wrapper = stagger.StaggerWrapper()

    assert wrapper is not None
    assert wrapper.stagger_jar_path == os.path.join(stagger_home, "stagger.jar")
    assert wrapper.stagger_model_path == os.path.join(stagger_home, "swedish.bin")

def test_create_wrapper_by_args():

    stagger_home = os.path.abspath('./lib/stagger')
    stagger_jar_path = os.path.join(stagger_home, "stagger.jar")
    stagger_model_path = os.path.join(stagger_home, "swedish.bin")

    wrapper = stagger.StaggerWrapper(stagger_jar_path, stagger_model_path)

    assert wrapper is not None
    assert wrapper.stagger_jar_path == os.path.join(stagger_home, "stagger.jar")
    assert wrapper.stagger_model_path == os.path.join(stagger_home, "swedish.bin")

def test_build_command():

    stagger_home = os.path.abspath('./lib/stagger')
    stagger_jar_path = os.path.join(stagger_home, "stagger.jar")
    stagger_model_path = os.path.join(stagger_home, "swedish.bin")

    wrapper = stagger.StaggerWrapper(stagger_jar_path, stagger_model_path)

    command = wrapper.build_command("*.txt", "16G")

    assert isinstance(command, str)
