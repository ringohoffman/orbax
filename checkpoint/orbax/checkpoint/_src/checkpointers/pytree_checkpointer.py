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

from typing import Optional
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.checkpointers import checkpointer
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler


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
    """Restores a PyTree.
    
    If `target` is provided, it automatically constructs the necessary `PyTreeRestoreArgs`
    to map the array shapes and shardings properly.
    """
    if target is not None:
      import jax
      from orbax.checkpoint import type_handlers

      def _get_restore_arg(x):
        if isinstance(x, jax.ShapeDtypeStruct) and getattr(x, 'sharding', None) is not None:
          from orbax.checkpoint import args as ocp_args
          return ocp_args.ArrayRestore(
              restore_args=type_handlers.ArrayRestoreArgs(
                  restore_type=jax.Array,
                  sharding=x.sharding,
                  global_shape=x.shape,
                  dtype=x.dtype,
              )
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
    return super().restore(directory, *args, **kwargs)
