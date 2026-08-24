# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import threading
import types
from typing import Any, ClassVar
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
from tokamax._src import benchmarking
from tokamax._src.autotuning import api
from tokamax._src.autotuning import artifact
from tokamax._src.autotuning import autotuner
from tokamax._src.ops import op as op_lib


@dataclasses.dataclass(frozen=True)
class _FakeConfig:
  value: int


class _FakeOp(op_lib.Op[Any, jax.Array, types.NoneType, _FakeConfig, Any]):
  config_cls: ClassVar[type[_FakeConfig]] = _FakeConfig

  def _fwd(self, x: jax.Array, *, return_residuals: bool, config):
    return x + config.value, None

  def _get_heuristics_config(self, ba: op_lib.BoundArguments) -> _FakeConfig:
    del ba
    return _FakeConfig(99)


@dataclasses.dataclass(frozen=True)
class _Capability:
  target_fd: int
  target_size: int
  target_sha256: str
  target_device: int
  target_inode: int
  target_uid: int
  target_gid: int
  target_mode: int
  target_nlink: int
  artifact_kind: str
  kind_metadata_schema: str
  identity_sha256: str
  receipt_sha256: str


def _digest(character: str) -> str:
  return character * 64


def _identity(
    device_kind: str, lowered: jax.stages.Lowered
) -> artifact.ExecutionIdentity:
  identity = artifact.execution_identity(
      tokamax_source_tree_sha256=_digest("1"),
      driver_runtime_manifest_sha256=_digest("2"),
      hardware_topology_manifest_sha256=_digest("3"),
      lowered=lowered,
  )
  if identity.device_kind != device_kind:
    raise RuntimeError("test device kind mismatch")
  return identity


def _result() -> tuple[
    api.AutotuningResult,
    op_lib.BoundArguments,
    _FakeOp,
    jax.stages.Lowered,
]:
  op = _FakeOp()
  shape = jax.ShapeDtypeStruct((2, 3), jnp.float32)
  bound_args = op.bind(shape)
  benchmark = benchmarking.BenchmarkData(
      compile_time_ms=1.0,
      lower_time_ms=2.0,
      evaluation_times_ms=(3.0, 4.0),
      metadata={},
      peak_memory_mb=5.0,
  )
  data = autotuner.AutotuningData({_FakeConfig(7): benchmark})
  device_kind = jax.devices()[0].device_kind
  result = api.AutotuningResult(device_kind, ((bound_args, data),))
  with result:
    lowered = jax.jit(op).lower(shape)
  return result, bound_args, op, lowered


def _publish(
    result: api.AutotuningResult,
    identity: artifact.ExecutionIdentity,
    lowered: jax.stages.Lowered,
    directory_fd: int,
    name: str,
) -> artifact.PublishedArtifact:
  return artifact.build_candidate(result, identity, lowered, directory_fd, name)


def _capability(
    fd: int,
    published: artifact.PublishedArtifact,
    *,
    kind: str = artifact.ARTIFACT_KIND,
) -> _Capability:
  observed = os.fstat(fd)
  return _Capability(
      target_fd=fd,
      target_size=observed.st_size,
      target_sha256=published.sha256,
      target_device=observed.st_dev,
      target_inode=observed.st_ino,
      target_uid=observed.st_uid,
      target_gid=observed.st_gid,
      target_mode=observed.st_mode & 0o7777,
      target_nlink=observed.st_nlink,
      artifact_kind=kind,
      kind_metadata_schema=artifact.KIND_METADATA_SCHEMA,
      identity_sha256=published.identity_sha256,
      receipt_sha256=_digest("5"),
  )


def _load_required(
    capability: _Capability, identity: artifact.ExecutionIdentity
) -> artifact.RequiredAutotuningCache:
  return artifact.load_required(capability, identity, _digest("5"))


class _PrimaryCompileError(BaseException):
  pass


def _exercise_corrupted_required_stack(failure_stage: str):
  result, _, op, lowered = _result()
  required = artifact.RequiredAutotuningCache(
      result,
      _identity(result.device_kind, lowered),
      artifact._LOAD_REQUIRED_TOKEN,  # pylint: disable=protected-access
  )
  state = op_lib.get_autotuning_cache_overlay_state()
  baseline = tuple(state.stack)
  corruption = object()

  def fail():
    state.stack.append(corruption)
    raise _PrimaryCompileError(f"{failure_stage} failed")

  try:
    if failure_stage == "lower":

      def fail_lower(_value):
        fail()

      required.compile(jax.jit(fail_lower), jnp.zeros((2, 3), jnp.float32))
    elif failure_stage == "compile":

      def fail_compile(_lowered):
        fail()

      with mock.patch.object(type(lowered), "compile", new=fail_compile):
        required.compile(jax.jit(op), jnp.zeros((2, 3), jnp.float32))
    else:
      raise ValueError(f"unknown failure stage: {failure_stage}")
  except BaseException as error:  # pylint: disable=broad-exception-caught
    observed = tuple(state.stack)
    state.stack[:] = baseline
    return error, observed, baseline
  raise AssertionError(f"{failure_stage} unexpectedly succeeded")


class ArtifactTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = Path(self.create_tempdir().full_path)
    self.directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)

  def tearDown(self):
    os.close(self.directory_fd)
    super().tearDown()

  def test_required_cache_cleanup_preserves_primary_in_normal_and_optimized_python(
      self,
  ):
    code = """
from tokamax._src.autotuning.artifact_test import _exercise_corrupted_required_stack
from tokamax._src.autotuning.artifact_test import _PrimaryCompileError
for stage in ("lower", "compile"):
  error, observed, baseline = _exercise_corrupted_required_stack(stage)
  if type(error) is not _PrimaryCompileError or str(error) != f"{stage} failed":
    raise AssertionError(f"primary error replaced for {stage}: {error!r}")
  if observed != baseline:
    raise AssertionError(f"stack leaked for {stage}: {observed!r}")
  notes = getattr(error, "__notes__", ())
  if len(notes) != 1 or "required autotuning cache stack is corrupted" not in notes[0]:
    raise AssertionError(f"cleanup error not aggregated for {stage}: {notes!r}")
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = str(Path(__file__).parents[3])
    for optimized in (False, True):
      command = [sys.executable]
      if optimized:
        command.append("-O")
      result = subprocess.run(
          [*command, "-c", code],
          check=False,
          env=environment,
          stdin=subprocess.DEVNULL,
          capture_output=True,
          text=True,
          timeout=90,
      )
      with self.subTest(optimized=optimized):
        self.assertEqual(result.returncode, 0, result.stderr)

  def test_publish_and_load_required_cache(self):
    result, bound_args, _, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    target = self.directory / "cache.json"
    self.assertEqual(target.stat().st_mode & 0o7777, 0o400)
    self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), published.sha256)

    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    try:
      os.lseek(fd, 1, os.SEEK_SET)
      required = _load_required(_capability(fd, published), identity)
      self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), 1)
      compiled = required.compile(
          jax.jit(bound_args.op), jnp.zeros((2, 3), jnp.float32)
      )
      output = compiled(jnp.zeros((2, 3), jnp.float32))
      self.assertEqual(jax.device_get(output).tolist(), [[7.0] * 3] * 2)
      self.assertEqual(bound_args.default_config, _FakeConfig(99))
    finally:
      os.close(fd)

  def test_builder_selects_one_fastest_valid_config(self):
    result, bound_args, _, lowered = _result()
    benchmark = next(iter(result.data[0][1].values()))
    slow = dataclasses.replace(benchmark, evaluation_times_ms=(20.0, 21.0))
    fast = dataclasses.replace(benchmark, evaluation_times_ms=(1.0, 2.0))
    result = dataclasses.replace(
        result,
        data=(
            (
                bound_args,
                autotuner.AutotuningData({
                    _FakeConfig(6): slow,
                    _FakeConfig(8): fast,
                    _FakeConfig(9): RuntimeError("unsupported"),
                }),
            ),
        ),
    )
    with result:
      lowered = jax.jit(bound_args.op).lower(
          jax.ShapeDtypeStruct((2, 3), jnp.float32)
      )
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    fd = os.open(self.directory / "cache.json", os.O_RDONLY | os.O_NOFOLLOW)
    try:
      required = _load_required(_capability(fd, published), identity)
      compiled = required.compile(
          jax.jit(bound_args.op), jnp.zeros((2, 3), jnp.float32)
      )
      output = compiled(jnp.zeros((2, 3), jnp.float32))
      self.assertEqual(jax.device_get(output).tolist(), [[8.0] * 3] * 2)
    finally:
      os.close(fd)

  def test_required_cache_miss_rejects_ambient_cache(self):
    result, _, op, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    missing = op.bind(jax.ShapeDtypeStruct((7, 11), jnp.float32))
    benchmark = next(iter(result.data[0][1].values()))
    ambient = autotuner.AutotuningData({_FakeConfig(8): benchmark})
    op.get_autotuning_cache(result.device_kind)[missing.autotuning_cache_key] = ambient
    self.assertEqual(missing.default_config, _FakeConfig(8))

    fd = os.open(self.directory / "cache.json", os.O_RDONLY | os.O_NOFOLLOW)
    try:
      required = _load_required(_capability(fd, published), identity)
      with self.assertRaisesRegex(ValueError, "Required autotuning cache miss"):
        required.compile(
            jax.jit(op), jnp.zeros((7, 11), jnp.float32)
        )

      def autotune_during_lowering(value):
        op.bind(value).autotune({_FakeConfig(9)})
        return value

      required = _load_required(_capability(fd, published), identity)
      with self.assertRaisesRegex(RuntimeError, "Autotuning is disabled"):
        required.compile(
            jax.jit(autotune_during_lowering),
            jnp.zeros((2, 3), jnp.float32),
        )

      def overlay_during_lowering(value):
        with result:
          return value

      required = _load_required(_capability(fd, published), identity)
      with self.assertRaisesRegex(RuntimeError, "Cannot overlay"):
        required.compile(
            jax.jit(overlay_during_lowering),
            jnp.zeros((2, 3), jnp.float32),
        )
    finally:
      os.close(fd)

  def test_publish_is_no_clobber(self):
    target = self.directory / "cache.json"
    target.write_bytes(b"existing")
    result, _, _, lowered = _result()
    with self.assertRaises(FileExistsError):
      _publish(
          result,
          _identity(result.device_kind, lowered),
          lowered,
          self.directory_fd,
          "cache.json",
      )
    self.assertEqual(target.read_bytes(), b"existing")
    self.assertEqual({path.name for path in self.directory.iterdir()}, {"cache.json"})

  def test_publish_fsyncs_file_and_directory_around_rename(self):
    result, _, _, lowered = _result()
    events = []
    original_fsync = artifact.os.fsync
    original_rename = artifact._rename_noreplace  # pylint: disable=protected-access

    def fsync(fd):
      role = "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
      events.append(f"fsync_{role}")
      original_fsync(fd)

    def rename(*args):
      events.append("rename_noreplace")
      original_rename(*args)

    with mock.patch.object(artifact.os, "fsync", side_effect=fsync), mock.patch.object(
        artifact, "_rename_noreplace", side_effect=rename
    ):
      _publish(
          result,
          _identity(result.device_kind, lowered),
          lowered,
          self.directory_fd,
          "cache.json",
      )
    self.assertEqual(
        events,
        [
            "fsync_file",
            "fsync_directory",
            "rename_noreplace",
            "fsync_directory",
        ],
    )

  def test_concurrent_publish_has_one_winner_and_no_temporary(self):
    result, _, _, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    barrier = threading.Barrier(2)
    original_rename = artifact._rename_noreplace  # pylint: disable=protected-access
    published = []
    errors = []

    def rename(*args):
      barrier.wait(timeout=10)
      original_rename(*args)

    def publish():
      try:
        published.append(
            _publish(
                result,
                identity,
                lowered,
                self.directory_fd,
                "cache.json",
            )
        )
      except BaseException as error:  # pylint: disable=broad-exception-caught
        errors.append(error)

    with mock.patch.object(artifact, "_rename_noreplace", side_effect=rename):
      threads = [threading.Thread(target=publish) for _ in range(2)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join(timeout=20)
    self.assertTrue(all(not thread.is_alive() for thread in threads))
    self.assertLen(published, 1)
    self.assertLen(errors, 1)
    self.assertIsInstance(errors[0], FileExistsError)
    self.assertEqual([path.name for path in self.directory.iterdir()], ["cache.json"])
    self.assertEqual(
        hashlib.sha256((self.directory / "cache.json").read_bytes()).hexdigest(),
        published[0].sha256,
    )

  def test_publish_rejects_path_output_name(self):
    result, _, _, lowered = _result()
    with self.assertRaisesRegex(ValueError, "ASCII .json component"):
      _publish(
          result,
          _identity(result.device_kind, lowered),
          lowered,
          self.directory_fd,
          "../cache.json",
      )
    self.assertEmpty(list(self.directory.iterdir()))

  def test_publish_uses_held_directory(self):
    held = self.directory.with_name(f"held-{secrets.token_hex(8)}")
    self.directory.rename(held)
    self.directory.mkdir()
    result, _, _, lowered = _result()
    _publish(
        result,
        _identity(result.device_kind, lowered),
        lowered,
        self.directory_fd,
        "cache.json",
    )
    self.assertTrue((held / "cache.json").is_file())
    self.assertFalse((self.directory / "cache.json").exists())

  def test_publish_failure_removes_owned_temporary(self):
    result, _, _, lowered = _result()
    with mock.patch.object(
        artifact, "_rename_noreplace", side_effect=OSError("injected")
    ):
      with self.assertRaisesRegex(OSError, "injected"):
        _publish(
            result,
            _identity(result.device_kind, lowered),
            lowered,
            self.directory_fd,
            "cache.json",
        )
    self.assertEmpty(list(self.directory.iterdir()))

  def test_publish_cleanup_failure_closes_held_directory(self):
    result, _, _, lowered = _result()
    duplicated = []
    closed = []
    original_fcntl = artifact.fcntl.fcntl
    original_close = artifact.os.close

    def fcntl_call(fd, command, *args):
      result_fd = original_fcntl(fd, command, *args)
      if command == artifact.fcntl.F_DUPFD_CLOEXEC:
        duplicated.append(result_fd)
      return result_fd

    def close(fd):
      closed.append(fd)
      original_close(fd)

    with mock.patch.object(
        artifact, "_rename_noreplace", side_effect=OSError("publish failed")
    ), mock.patch.object(
        artifact, "_unlink_owned_name", side_effect=OSError("cleanup failed")
    ), mock.patch.object(
        artifact.fcntl, "fcntl", side_effect=fcntl_call
    ), mock.patch.object(
        artifact.os, "close", side_effect=close
    ):
      with self.assertRaisesRegex(OSError, "cleanup failed"):
        _publish(
            result,
            _identity(result.device_kind, lowered),
            lowered,
            self.directory_fd,
            "cache.json",
        )
    self.assertLen(duplicated, 1)
    self.assertIn(duplicated[0], closed)

  def test_publish_write_failure_removes_owned_temporary(self):
    result, _, _, lowered = _result()
    with mock.patch.object(artifact.os, "write", side_effect=OSError("injected")):
      with self.assertRaisesRegex(OSError, "injected"):
        _publish(
            result,
            _identity(result.device_kind, lowered),
            lowered,
            self.directory_fd,
            "cache.json",
        )
    self.assertEmpty(list(self.directory.iterdir()))

  def test_publish_readback_failure_removes_final_name(self):
    result, _, _, lowered = _result()
    with mock.patch.object(
        artifact, "_pread_exact", side_effect=OSError("injected")
    ):
      with self.assertRaisesRegex(OSError, "injected"):
        _publish(
            result,
            _identity(result.device_kind, lowered),
            lowered,
            self.directory_fd,
            "cache.json",
        )
    self.assertEmpty(list(self.directory.iterdir()))

  def test_publish_rejects_lowering_with_different_config(self):
    result, _, op, _ = _result()
    lowered = jax.jit(op).lower(
        jax.ShapeDtypeStruct((2, 3), jnp.float32)
    )
    identity = _identity(result.device_kind, lowered)
    with self.assertRaisesRegex(ValueError, "configured lowered program"):
      artifact.build_candidate(
          result, identity, lowered, self.directory_fd, "cache.json"
      )
    self.assertEmpty(list(self.directory.iterdir()))

  def test_required_cache_rejects_sealed_config_identity_mismatch(self):
    result, _, op, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    wrong_lowered = jax.jit(op).lower(
        jax.ShapeDtypeStruct((2, 3), jnp.float32)
    )
    wrong_identity = dataclasses.replace(
        identity,
        lowered_program_sha256=artifact.lowered_program_sha256(wrong_lowered),
    )
    target = self.directory / "cache.json"
    value = artifact._strict_json(target.read_bytes())  # pylint: disable=protected-access
    value["identity"] = wrong_identity.as_dict()
    value["identity_sha256"] = wrong_identity.sha256
    data = artifact._canonical_json(value)  # pylint: disable=protected-access
    target.chmod(0o600)
    target.write_bytes(data)
    target.chmod(0o400)
    forged = artifact.PublishedArtifact(
        len(data), hashlib.sha256(data).hexdigest(), wrong_identity.sha256
    )
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    try:
      required = _load_required(_capability(fd, forged), wrong_identity)
      with self.assertRaisesRegex(ValueError, "cache identity"):
        required.compile(
            jax.jit(op), jnp.zeros((2, 3), jnp.float32)
        )
    finally:
      os.close(fd)

  def test_load_uses_held_file_after_path_swap(self):
    result, _, _, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    target = self.directory / "cache.json"
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    try:
      target.rename(self.directory / "held.json")
      target.write_bytes(b"replacement")
      required = _load_required(_capability(fd, published), identity)
      required.compile(
          jax.jit(result.data[0][0].op), jnp.zeros((2, 3), jnp.float32)
      )
    finally:
      os.close(fd)

  def test_load_rejects_wrong_identity(self):
    result, _, _, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    wrong = dataclasses.replace(identity, lowered_program_sha256=_digest("6"))
    fd = os.open(self.directory / "cache.json", os.O_RDONLY | os.O_NOFOLLOW)
    try:
      with self.assertRaisesRegex(ValueError, "execution identity mismatch"):
        _load_required(_capability(fd, published), wrong)
    finally:
      os.close(fd)

  def test_required_cache_rejects_mixed_live_program(self):
    result, _, op, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    fd = os.open(self.directory / "cache.json", os.O_RDONLY | os.O_NOFOLLOW)
    try:
      required = _load_required(_capability(fd, published), identity)
      with self.assertRaisesRegex(TypeError, "exact jax.stages.Wrapped"):
        required.compile(lowered)
      with self.assertRaisesRegex(ValueError, "cache identity"):
        required.compile(
            jax.jit(lambda value: op(value) + 2),
            jnp.zeros((2, 3), jnp.float32),
        )
    finally:
      os.close(fd)

  def test_required_cache_rejects_explicit_config_bypass(self):
    result, _, op, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    fd = os.open(self.directory / "cache.json", os.O_RDONLY | os.O_NOFOLLOW)
    try:
      required = _load_required(_capability(fd, published), identity)
      configured = op.replace(config=_FakeConfig(7))
      with self.assertRaisesRegex(ValueError, "Required autotuning cache miss"):
        required.compile(
            jax.jit(configured), jnp.zeros((2, 3), jnp.float32)
        )
    finally:
      os.close(fd)

  def test_required_cache_rejects_unused_sealed_entry(self):
    result, _, op, _ = _result()
    second_shape = jax.ShapeDtypeStruct((7, 11), jnp.float32)
    second_bound_args = op.bind(second_shape)
    tuning_data = result.data[0][1]
    result = dataclasses.replace(
        result, data=(*result.data, (second_bound_args, tuning_data))
    )
    with result:
      lowered = jax.jit(lambda first, second: (op(first), op(second))).lower(
          jax.ShapeDtypeStruct((2, 3), jnp.float32), second_shape
      )
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    fd = os.open(self.directory / "cache.json", os.O_RDONLY | os.O_NOFOLLOW)
    try:
      required = _load_required(_capability(fd, published), identity)
      with self.assertRaisesRegex(ValueError, "every entry"):
        required.compile(
            jax.jit(op), jnp.zeros((2, 3), jnp.float32)
        )
    finally:
      os.close(fd)

  def test_required_cache_is_single_use_nested_safe_and_does_not_escape(self):
    result, bound_args, op, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    fd = os.open(self.directory / "cache.json", os.O_RDONLY | os.O_NOFOLLOW)
    try:
      outer = _load_required(_capability(fd, published), identity)
      nested = _load_required(_capability(fd, published), identity)

      def enter_nested(value):
        nested.compile(jax.jit(op), value)
        return op(value)

      with self.assertRaisesRegex(RuntimeError, "another required"):
        outer.compile(
            jax.jit(enter_nested), jnp.zeros((2, 3), jnp.float32)
        )
      self.assertEmpty(op_lib.get_autotuning_cache_overlay_state().stack)

      required = _load_required(_capability(fd, published), identity)
      with self.assertRaises(TypeError):
        with required:  # type: ignore[attr-defined]
          pass
      compiled = required.compile(
          jax.jit(op), jnp.zeros((2, 3), jnp.float32)
      )
      output = compiled(jnp.zeros((2, 3), jnp.float32))
      self.assertEqual(jax.device_get(output).tolist(), [[7.0] * 3] * 2)
      self.assertEqual(bound_args.default_config, _FakeConfig(99))
      self.assertEmpty(op_lib.get_autotuning_cache_overlay_state().stack)
      with self.assertRaisesRegex(RuntimeError, "already consumed"):
        required.compile(
            jax.jit(op), jnp.zeros((2, 3), jnp.float32)
        )
      self.assertFalse(hasattr(required, "verify_lowered"))
    finally:
      os.close(fd)

  def test_load_rejects_writable_fd(self):
    result, _, _, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    target = self.directory / "cache.json"
    target.chmod(0o600)
    fd = os.open(target, os.O_RDWR | os.O_NOFOLLOW)
    try:
      capability = dataclasses.replace(
          _capability(fd, published), target_mode=0o400
      )
      with self.assertRaisesRegex(ValueError, "not read-only"):
        _load_required(capability, identity)
    finally:
      os.close(fd)

  def test_load_rejects_inheritable_fd(self):
    result, _, _, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    fd = os.open(self.directory / "cache.json", os.O_RDONLY | os.O_NOFOLLOW)
    try:
      os.set_inheritable(fd, True)
      with self.assertRaisesRegex(ValueError, "not close-on-exec"):
        _load_required(_capability(fd, published), identity)
    finally:
      os.close(fd)

  def test_load_rejects_wrong_registrar_kind(self):
    result, _, _, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    fd = os.open(self.directory / "cache.json", os.O_RDONLY | os.O_NOFOLLOW)
    try:
      with self.assertRaisesRegex(ValueError, "wrong artifact kind"):
        _load_required(_capability(fd, published, kind="model/v1"), identity)
    finally:
      os.close(fd)

  def test_load_rejects_wrong_artifact_receipt(self):
    result, _, _, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    fd = os.open(self.directory / "cache.json", os.O_RDONLY | os.O_NOFOLLOW)
    try:
      capability = dataclasses.replace(
          _capability(fd, published), receipt_sha256=_digest("6")
      )
      with self.assertRaisesRegex(ValueError, "wrong artifact receipt"):
        _load_required(capability, identity)
    finally:
      os.close(fd)

  def test_load_rejects_multiply_linked_target(self):
    result, _, _, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    target = self.directory / "cache.json"
    os.link(target, self.directory / "alias.json")
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    try:
      with self.assertRaisesRegex(ValueError, "sealed file"):
        _load_required(_capability(fd, published), identity)
    finally:
      os.close(fd)

  def test_load_rejects_noncanonical_artifact(self):
    result, _, _, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    published = _publish(
        result, identity, lowered, self.directory_fd, "cache.json"
    )
    target = self.directory / "cache.json"
    value = json.loads(target.read_text())
    target.chmod(0o600)
    target.write_text(json.dumps(value, indent=2))
    target.chmod(0o400)
    data = target.read_bytes()
    forged = dataclasses.replace(
        published, size=len(data), sha256=hashlib.sha256(data).hexdigest()
    )
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    try:
      with self.assertRaisesRegex(ValueError, "encoding is not canonical"):
        _load_required(_capability(fd, forged), identity)
    finally:
      os.close(fd)

  def test_strict_json_rejects_duplicate_and_nonfinite_values(self):
    with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
      artifact._strict_json('{"value":1,"value":2}')  # pylint: disable=protected-access
    with self.assertRaisesRegex(ValueError, "non-finite"):
      artifact._strict_json('{"value":NaN}')  # pylint: disable=protected-access
    with self.assertRaisesRegex(ValueError, "non-finite"):
      artifact._strict_json('{"value":1e400}')  # pylint: disable=protected-access

  def test_identity_rejects_bool_count_and_placeholder(self):
    result, _, _, lowered = _result()
    identity = _identity(result.device_kind, lowered)
    with self.assertRaisesRegex(ValueError, "positive integer"):
      dataclasses.replace(identity, jax_process_count=True)
    with self.assertRaisesRegex(ValueError, "placeholder"):
      dataclasses.replace(identity, lowered_program_sha256="0" * 64)

  def test_lowered_program_digest_binds_shape(self):
    def add_one(value):
      return value + 1

    first = jax.jit(add_one).lower(
        jax.ShapeDtypeStruct((2, 3), jnp.float32)
    )
    repeated = jax.jit(add_one).lower(
        jax.ShapeDtypeStruct((2, 3), jnp.float32)
    )
    second = jax.jit(add_one).lower(
        jax.ShapeDtypeStruct((3, 2), jnp.float32)
    )
    self.assertEqual(
        artifact.lowered_program_sha256(first),
        artifact.lowered_program_sha256(repeated),
    )
    self.assertNotEqual(
        artifact.lowered_program_sha256(first),
        artifact.lowered_program_sha256(second),
    )
    with self.assertRaisesRegex(TypeError, "exact jax.stages.Lowered"):
      artifact.lowered_program_sha256(object())

  def test_result_rejects_nonfinite_benchmark(self):
    result, _, _, lowered = _result()
    bound_args, data = result.data[0]
    config = next(iter(data))
    invalid = dataclasses.replace(
        data[config], evaluation_times_ms=(float("nan"),)
    )
    result = dataclasses.replace(
        result,
        data=((bound_args, autotuner.AutotuningData({config: invalid})),),
    )
    with self.assertRaisesRegex(ValueError, "invalid measurements"):
      _publish(
          result,
          _identity(result.device_kind, lowered),
          lowered,
          self.directory_fd,
          "cache.json",
      )


if __name__ == "__main__":
  absltest.main()
