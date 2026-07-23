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

"""Shorthand for `Checkpointer(PyTreeCheckpointHandler())`."""

import contextlib
import logging
from typing import Optional

from etils import epath
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.checkpointers import checkpointer
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler


_logger = logging.getLogger(__name__)


def _is_ocdbt_checkpoint(directory: str) -> bool:
  """Returns True if `directory` contains an OCDBT manifest."""
  path = epath.Path(str(directory))
  return (path / 'manifest.ocdbt').exists()


@contextlib.contextmanager
def _use_standard_array_handler():
  """Temporarily use the standard ArrayHandler for ``jax.Array``.

  When a custom handler (e.g. ``CloudPathwaysArrayHandler`` from
  ``pathwaysutils``) is registered globally, it intercepts all
  ``jax.Array`` restore operations.  ``CloudPathwaysArrayHandler`` uses
  the Pathways Persistence API which constructs TensorStore specs with
  ``"driver": "zarr"`` — it does **not** support the OCDBT TensorStore
  driver.

  For OCDBT-format checkpoints we must restore via the standard
  ``ArrayHandler`` which reads through TensorStore (supporting both
  OCDBT and plain zarr).  For non-OCDBT checkpoints the custom handler
  is preferred because it enables direct GCS-to-HBM DMA on Pathways
  TPU workers (zero host/proxy memory).

  This context manager:
    1. Saves the currently registered ``jax.Array`` handler.
    2. Replaces it with a fresh ``ArrayHandler``.
    3. Restores the original handler on exit.

  If the current handler is already the standard ``ArrayHandler``,
  this is a no-op.
  """
  import jax
  from orbax.checkpoint import type_handlers
  from orbax.checkpoint._src.serialization import jax_array_handlers

  current_handler = type_handlers.get_type_handler(jax.Array)
  is_standard = type(current_handler) is jax_array_handlers.ArrayHandler

  if is_standard:
    yield
    return

  _logger.info(
      'OCDBT checkpoint detected — temporarily using standard Orbax '
      'ArrayHandler for restore (registered handler %s does not '
      'support OCDBT).  Will restore %s after loading.',
      type(current_handler).__name__,
      type(current_handler).__name__,
  )
  type_handlers.register_type_handler(
      jax.Array, jax_array_handlers.ArrayHandler(), override=True
  )
  try:
    yield
  finally:
    type_handlers.register_type_handler(
        jax.Array, current_handler, override=True
    )
    _logger.info(
        'Restored %s as jax.Array handler after OCDBT checkpoint load.',
        type(current_handler).__name__,
    )


class PyTreeCheckpointer(checkpointer.Checkpointer):
  """Shorthand class.

  Instead of::
    ckptr = Checkpointer(PyTreeCheckpointHandler())

  we can use::
    ckptr = PyTreeCheckpointer()
  """

  def __init__(
      self,
      primary_host: Optional[int] = 0,
      use_ocdbt: bool = True,
      use_zarr3=False,
      use_compression: bool = True,
  ):
    super().__init__(
        pytree_checkpoint_handler.PyTreeCheckpointHandler(
            use_ocdbt=use_ocdbt,
            use_zarr3=use_zarr3,
            use_compression=use_compression,
        ),
        multiprocessing_options=options_lib.MultiprocessingOptions(
            primary_host=primary_host
        ),
    )

  def restore(
      self,
      directory,
      *args,
      target=None,
      partial_restore: bool = False,
      **kwargs,
  ):
    """Restores a PyTree with automatic format detection.

    If ``target`` is provided (a tree of ``jax.ShapeDtypeStruct`` with
    shardings), this method automatically constructs the necessary
    ``PyTreeRestoreArgs`` and dispatches to the most efficient
    compatible handler:

    * **OCDBT checkpoints** (``manifest.ocdbt`` present): uses the
      standard Orbax ``ArrayHandler`` via TensorStore.  This is
      required because ``CloudPathwaysArrayHandler`` (from
      ``pathwaysutils``) does not support the OCDBT TensorStore driver.

    * **Zarr checkpoints** (per-tensor ``.zarray`` files): uses
      whichever handler is globally registered.  When
      ``CloudPathwaysArrayHandler`` is active, this enables direct
      GCS-to-HBM DMA on Pathways TPU workers with zero host memory.

    Args:
      directory: Checkpoint directory path.
      target: Optional tree of ``jax.ShapeDtypeStruct`` describing the
        desired output shapes and shardings.
      partial_restore: If True, allow restoring a subset of the
        checkpoint tree.
      **kwargs: Forwarded to ``Checkpointer.restore()``.
    """
    if target is not None:
      import jax
      from orbax.checkpoint import type_handlers

      def _get_restore_arg(x):
        if isinstance(x, jax.ShapeDtypeStruct) and getattr(x, 'sharding', None) is not None:
          return type_handlers.ArrayRestoreArgs(
              restore_type=jax.Array,
              sharding=x.sharding,
              global_shape=x.shape,
              dtype=x.dtype,
          )
        return None

      restore_args = jax.tree_util.tree_map(
          _get_restore_arg,
          target,
          is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct)
      )

      if 'args' not in kwargs:
        kwargs['args'] = pytree_checkpoint_handler.PyTreeRestoreArgs(
            item=target,
            restore_args=restore_args,
            partial_restore=partial_restore,
        )

      with (
          _use_standard_array_handler()
          if _is_ocdbt_checkpoint(directory)
          else contextlib.nullcontext()
      ):
        return super().restore(directory, *args, **kwargs)

    return super().restore(directory, *args, **kwargs)
