# Copyright 2026 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""To test Orbax in single-host setup."""

import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.path import deleter as deleter_lib
from orbax.checkpoint._src.path import step as step_lib


class CheckpointDeleterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = epath.Path(self.create_tempdir('ckpt').full_path)

  def _get_save_diretory(self, step: int, directory: epath.Path) -> epath.Path:
    return directory / str(step)

  @parameterized.product(
      threaded=(False, True),
      distributed=(False, True),
      todelete_subdir=(None, 'some_delete_dir'),
  )
  def test_checkpoint_deleter_delete(
      self, threaded, distributed, todelete_subdir
  ):
    """Test regular, threaded, and distributed CheckpointDeleter."""
    if threaded and distributed:
      return  # Mutually exclusive options in test matrix
    deleter = deleter_lib.create_checkpoint_deleter(
        self.ckpt_dir,
        name_format=step_lib.standard_name_format(),
        primary_host=None,
        todelete_subdir=todelete_subdir,
        todelete_full_path=None,
        enable_background_delete=threaded,
        enable_distributed_delete=distributed,
    )

    step = 1
    step_dir = self._get_save_diretory(step, self.ckpt_dir)
    step_dir.mkdir()
    (step_dir / 'item1').mkdir()
    (step_dir / 'item1' / 'tensor1').mkdir()
    self.assertTrue(step_dir.exists())
    deleter.delete(step)
    deleter.close()

    # assert the step_dir is deleted
    self.assertFalse(step_dir.exists())

    # In case of rename, check if the new folder exists
    if todelete_subdir is not None:
      self.assertTrue((self.ckpt_dir / todelete_subdir / str(step)).exists())

    deleter.close()

  @mock.patch('orbax.checkpoint._src.multihost.multihost.sync_global_processes')
  @mock.patch('orbax.checkpoint._src.multihost.multihost.process_count', return_value=2)
  def test_distributed_checkpoint_deleter_multihost_sharding(
      self, mock_proc_count, mock_sync
  ):
    """Test multi-host sharding logic in DistributedCheckpointDeleter."""
    deleter = deleter_lib.DistributedCheckpointDeleter(
        self.ckpt_dir,
        name_format=step_lib.standard_name_format(),
        primary_host=0,
    )
    step = 42
    step_dir = self._get_save_diretory(step, self.ckpt_dir)
    step_dir.mkdir()
    
    # Create 4 tensor subdirectories
    subdirs = [
        step_dir / 'item1' / 't0',
        step_dir / 'item1' / 't1',
        step_dir / 'item2' / 't2',
        step_dir / 'item2' / 't3',
    ]
    for s in subdirs:
      s.mkdir(parents=True, exist_ok=True)
      (s / 'data.bin').write_text('dummy')

    all_subpaths = sorted([p for p in step_dir.glob('*/*')])
    # Process 0 deletes index 0, 2
    p0_subpaths = [p for idx, p in enumerate(all_subpaths) if idx % 2 == 0]
    # Process 1 deletes index 1, 3
    p1_subpaths = [p for idx, p in enumerate(all_subpaths) if idx % 2 == 1]

    for p in p0_subpaths:
      deleter._rmtree(p)

    self.assertFalse((step_dir / 'item1' / 't0').exists())
    self.assertTrue((step_dir / 'item1' / 't1').exists())
    self.assertFalse((step_dir / 'item2' / 't2').exists())
    self.assertTrue((step_dir / 'item2' / 't3').exists())

    for p in p1_subpaths:
      deleter._rmtree(p)

    self.assertFalse((step_dir / 'item1' / 't1').exists())
    self.assertFalse((step_dir / 'item2' / 't3').exists())

    # Primary host cleanup
    with mock.patch('orbax.checkpoint._src.multihost.multihost.process_index', return_value=0), \
         mock.patch('orbax.checkpoint._src.multihost.multihost.is_primary_host', return_value=True):
      deleter.delete(step)

    self.assertFalse(step_dir.exists())


class GcsRenameTest(unittest.TestCase):

  @mock.patch('orbax.checkpoint._src.path.deleter.epath.Path')
  def test_gcs_rename_logic_directly(self, mock_epath_constructor):
    """Tests path construction and rename call logic."""
    standard_checkpoint_deleter = deleter_lib.StandardCheckpointDeleter

    deleter = standard_checkpoint_deleter(
        directory=mock.MagicMock(),
        name_format=step_lib.standard_name_format(),
        primary_host=None,
        todelete_subdir=None,
        todelete_full_path='trash_bin',
    )
    # When epath.Path() is called inside the code, it returns this mock parent
    mock_dest_parent = mock.MagicMock()
    mock_epath_constructor.return_value = mock_dest_parent

    # When the code does (parent / child), return a specific final mock
    mock_final_dest = mock.MagicMock()
    mock_final_dest.__str__.return_value = 'gs://mocked/final/destination'
    mock_dest_parent.__truediv__.return_value = mock_final_dest

    # Setup the "Source" Mock (The step being deleted)
    mock_step_path = mock.MagicMock()
    mock_step_path.__str__.return_value = 'gs://my-bucket/checkpoints/step_10'
    mock_step_path.name = 'step_10'

    deleter._gcs_rename_step(step=10, delete_target=mock_step_path)

    # Verify mkdir was called on the destination parent.
    mock_dest_parent.mkdir.assert_called_with(parents=True, exist_ok=True)

    # Verify the Parent Path string was constructed correctly
    # The code does: epath.Path(f'gs://{bucket}/{todelete_full_path}')
    (parent_path_arg,), _ = mock_epath_constructor.call_args
    self.assertEqual(parent_path_arg, 'gs://my-bucket/trash_bin')

    # Verify the Child Filename was constructed correctly
    (child_name_arg,), _ = mock_dest_parent.__truediv__.call_args
    self.assertIn('step_10-', child_name_arg)

    # Verify the Rename was actually called
    mock_step_path.rename.assert_called_with(mock_final_dest)

if __name__ == '__main__':
  absltest.main()
