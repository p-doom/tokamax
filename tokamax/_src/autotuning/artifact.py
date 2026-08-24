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
"""Immutable autotuning-cache artifacts."""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
from typing import Any, Final, Protocol, Self

import jax
from tokamax._src import benchmarking
from tokamax._src.autotuning import api
from tokamax._src.autotuning import autotuner
from tokamax._src.ops import op as op_lib


ARTIFACT_KIND: Final[str] = "tokamax_autotuning_cache/v1"
KIND_METADATA_SCHEMA: Final[str] = "tokamax_autotuning_cache_identity/v1"
_IDENTITY_SCHEMA: Final[str] = "tokamax_autotuning_execution_identity/v1"
_PAYLOAD_SCHEMA: Final[str] = "tokamax_autotuning_result/json/v1"
_MAX_ARTIFACT_BYTES: Final[int] = 64 * 1024 * 1024
_DIGEST_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_DEVICE_KIND_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9 ._:+/-]{0,127}"
)
_OUTPUT_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json"
)
_RENAME_NOREPLACE: Final[int] = 1


def _require_digest(name: str, value: Any) -> str:
  if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
    raise ValueError(f"{name} must be a lowercase SHA-256 digest")
  if value == "0" * 64:
    raise ValueError(f"{name} must not be a placeholder digest")
  return value


def _require_positive_int(name: str, value: Any) -> int:
  if type(value) is not int or value <= 0:  # pylint: disable=unidiomatic-typecheck
    raise ValueError(f"{name} must be a positive integer")
  return value


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionIdentity:
  """Exact environment and lowered-program identity for one cache."""

  tokamax_source_tree_sha256: str
  driver_runtime_manifest_sha256: str
  hardware_topology_manifest_sha256: str
  lowered_program_sha256: str
  device_kind: str
  jax_process_count: int
  jax_device_count: int
  jax_local_device_count: int

  def __post_init__(self):
    for name in (
        "tokamax_source_tree_sha256",
        "driver_runtime_manifest_sha256",
        "hardware_topology_manifest_sha256",
        "lowered_program_sha256",
    ):
      _require_digest(name, getattr(self, name))
    if not isinstance(self.device_kind, str) or _DEVICE_KIND_RE.fullmatch(
        self.device_kind
    ) is None:
      raise ValueError("device_kind is invalid")
    process_count = _require_positive_int(
        "jax_process_count", self.jax_process_count
    )
    device_count = _require_positive_int("jax_device_count", self.jax_device_count)
    local_device_count = _require_positive_int(
        "jax_local_device_count", self.jax_local_device_count
    )
    if process_count > device_count:
      raise ValueError("jax_process_count exceeds jax_device_count")
    if local_device_count > device_count:
      raise ValueError("jax_local_device_count exceeds jax_device_count")

  def as_dict(self) -> dict[str, Any]:
    return {
        "schema": _IDENTITY_SCHEMA,
        "tokamax_source_tree_sha256": self.tokamax_source_tree_sha256,
        "driver_runtime_manifest_sha256": self.driver_runtime_manifest_sha256,
        "hardware_topology_manifest_sha256": self.hardware_topology_manifest_sha256,
        "lowered_program_sha256": self.lowered_program_sha256,
        "device_kind": self.device_kind,
        "jax_process_count": self.jax_process_count,
        "jax_device_count": self.jax_device_count,
        "jax_local_device_count": self.jax_local_device_count,
    }

  @classmethod
  def from_dict(cls, value: Any) -> Self:
    if type(value) is not dict:  # pylint: disable=unidiomatic-typecheck
      raise ValueError("identity must be an object")
    expected = {
        "schema",
        "tokamax_source_tree_sha256",
        "driver_runtime_manifest_sha256",
        "hardware_topology_manifest_sha256",
        "lowered_program_sha256",
        "device_kind",
        "jax_process_count",
        "jax_device_count",
        "jax_local_device_count",
    }
    if set(value) != expected:
      raise ValueError("identity has missing or unknown fields")
    if value["schema"] != _IDENTITY_SCHEMA:
      raise ValueError("identity schema is unsupported")
    return cls(**{name: value[name] for name in expected - {"schema"}})

  @property
  def sha256(self) -> str:
    return _sha256(_canonical_json(self.as_dict()))


class RegistrarReadCapability(Protocol):
  """Typed output of the node-local registrar capability client."""

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


@dataclasses.dataclass(frozen=True, slots=True)
class PublishedArtifact:
  size: int
  sha256: str
  identity_sha256: str


def _reject_constant(value: str) -> None:
  raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  result = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate JSON key: {key}")
    result[key] = value
  return result


def _strict_json(data: bytes | str) -> Any:
  if isinstance(data, bytes):
    text = data.decode("utf-8", errors="strict")
  elif isinstance(data, str):
    text = data
  else:
    raise TypeError("JSON input must be bytes or text")
  return json.loads(
      text,
      object_pairs_hook=_object_without_duplicates,
      parse_constant=_reject_constant,
  )


def _canonical_json(value: Any) -> bytes:
  return json.dumps(
      value,
      allow_nan=False,
      ensure_ascii=True,
      separators=(",", ":"),
      sort_keys=True,
  ).encode("ascii")


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _validate_benchmark(value: Any) -> None:
  if not isinstance(value, benchmarking.BenchmarkData):
    raise TypeError("autotuning cache contains a failed benchmark")
  if value.peak_memory_mb is None:
    raise ValueError("autotuning benchmark has no peak-memory measurement")
  if type(value.evaluation_times_ms) is not tuple:  # pylint: disable=unidiomatic-typecheck
    raise ValueError("autotuning benchmark samples must be a tuple")
  numeric_values = (
      value.compile_time_ms,
      value.lower_time_ms,
      *value.evaluation_times_ms,
      value.peak_memory_mb,
  )
  if not value.evaluation_times_ms:
    raise ValueError("autotuning benchmark has no evaluation samples")
  if any(  # pylint: disable=unidiomatic-typecheck
      type(number) is not float
      or not math.isfinite(number)
      or number < 0
      for number in numeric_values
  ):
    raise ValueError("autotuning benchmark contains invalid measurements")
  if type(value.metadata) is not dict or value.metadata:  # pylint: disable=unidiomatic-typecheck
    raise ValueError("autotuning benchmark metadata must be empty")


def _config_key(
    device_kind: str,
    bound_args: op_lib.BoundArguments,
    config: Any,
    benchmark: benchmarking.BenchmarkData,
) -> bytes:
  entry = api.AutotuningResult(
      device_kind,
      ((bound_args, autotuner.AutotuningData({config: benchmark})),),
  )
  value = _strict_json(entry.dumps())
  config_map = value["data"][0][1]
  if type(config_map) is not dict or len(config_map) != 1:  # pylint: disable=unidiomatic-typecheck
    raise ValueError("autotuning config encoding is invalid")
  return _canonical_json(next(iter(config_map)))


def _select_result(result: api.AutotuningResult) -> api.AutotuningResult:
  if not isinstance(result, api.AutotuningResult):
    raise TypeError("result must be an AutotuningResult")
  selected = []
  for bound_args, data in result.data:
    valid = []
    for config, benchmark in data.items():
      if not isinstance(benchmark, benchmarking.BenchmarkData):
        continue
      _validate_benchmark(benchmark)
      valid.append((config, benchmark))
    if not valid:
      raise ValueError("autotuning result has no valid config for an operation")
    config, benchmark = min(
        valid,
        key=lambda item: (
            item[1].median_evaluation_time_ms,
            _config_key(result.device_kind, bound_args, *item),
        ),
    )
    selected.append(
        (bound_args, autotuner.AutotuningData({config: benchmark}))
    )
  return api.AutotuningResult(result.device_kind, tuple(selected))


def _canonical_result(result: api.AutotuningResult) -> str:
  if not isinstance(result, api.AutotuningResult):
    raise TypeError("result must be an AutotuningResult")
  if not result.data:
    raise ValueError("autotuning result is empty")
  cache_keys = set()
  for bound_args, data in result.data:
    if not isinstance(data, autotuner.AutotuningData) or not data:
      raise ValueError("autotuning result contains an empty operation")
    for benchmark in data.values():
      _validate_benchmark(benchmark)
    entry = api.AutotuningResult(result.device_kind, ((bound_args, data),))
    entry_value = _strict_json(entry.dumps())
    cache_key = _canonical_json(entry_value["data"][0][0])
    if cache_key in cache_keys:
      raise ValueError("autotuning result contains a duplicate cache key")
    cache_keys.add(cache_key)
  payload = result.dumps()
  _strict_json(payload)
  if api.AutotuningResult.loads(payload).dumps() != payload:
    raise ValueError("autotuning result encoding is not canonical")
  return payload


def _bound_args_key(bound_args: op_lib.BoundArguments) -> bytes:
  normalized = bound_args.replace(
      op=bound_args.op.replace(config=None, vjp=None)
  )
  value = _strict_json(op_lib.BOUND_ARGS_ADAPTER.dump_json(normalized))
  del value["op"]["config"]
  del value["op"]["vjp"]
  return _canonical_json(value)


def _artifact_bytes(
    result: api.AutotuningResult,
    identity: ExecutionIdentity,
    lowered: jax.stages.Lowered,
) -> bytes:
  result = _select_result(result)
  if lowered_program_sha256(lowered) != identity.lowered_program_sha256:
    raise ValueError("lowered program does not match the execution identity")
  if result.device_kind != identity.device_kind:
    raise ValueError(
        f"result device kind {result.device_kind!r} does not match "
        f"identity {identity.device_kind!r}"
    )
  payload = _canonical_result(result)
  result_keys = {_bound_args_key(bound_args) for bound_args, _ in result.data}
  lowered_keys = {
      _bound_args_key(bound_args) for bound_args in api.get_bound_args(lowered)
  }
  if result_keys != lowered_keys:
    raise ValueError("autotuning result does not exactly cover the lowered program")
  return _canonical_json({
      "schema": ARTIFACT_KIND,
      "identity": identity.as_dict(),
      "identity_sha256": identity.sha256,
      "payload_schema": _PAYLOAD_SCHEMA,
      "payload_sha256": _sha256(payload.encode("utf-8")),
      "payload": payload,
  })


def lowered_program_sha256(lowered: Any) -> str:
  """Hashes the exact lowered program, including shape and sharding metadata."""
  if not isinstance(lowered, jax.stages.Lowered):
    raise TypeError("lowered must be an exact jax.stages.Lowered")
  text = lowered.as_text(debug_info=False)
  if not isinstance(text, str) or not text:
    raise ValueError("lowered program text is empty")
  return _sha256(text.encode("utf-8"))


def execution_identity(
    *,
    tokamax_source_tree_sha256: str,
    driver_runtime_manifest_sha256: str,
    hardware_topology_manifest_sha256: str,
    lowered: jax.stages.Lowered,
) -> ExecutionIdentity:
  devices = tuple(jax.devices())
  local_devices = tuple(jax.local_devices())
  device_kinds = {device.device_kind for device in devices}
  if not devices or not local_devices or len(device_kinds) != 1:
    raise RuntimeError("JAX device topology is empty or heterogeneous")
  return ExecutionIdentity(
      tokamax_source_tree_sha256=tokamax_source_tree_sha256,
      driver_runtime_manifest_sha256=driver_runtime_manifest_sha256,
      hardware_topology_manifest_sha256=hardware_topology_manifest_sha256,
      lowered_program_sha256=lowered_program_sha256(lowered),
      device_kind=device_kinds.pop(),
      jax_process_count=jax.process_count(),
      jax_device_count=len(devices),
      jax_local_device_count=len(local_devices),
  )


def _verify_live_environment(identity: ExecutionIdentity) -> None:
  devices = tuple(jax.devices())
  local_devices = tuple(jax.local_devices())
  observed = (
      {device.device_kind for device in devices},
      jax.process_count(),
      len(devices),
      len(local_devices),
  )
  expected = (
      {identity.device_kind},
      identity.jax_process_count,
      identity.jax_device_count,
      identity.jax_local_device_count,
  )
  if observed != expected:
    raise ValueError("live JAX hardware topology does not match the cache identity")


def _rename_noreplace(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
  libc = ctypes.CDLL(None, use_errno=True)
  renameat2 = getattr(libc, "renameat2", None)
  if renameat2 is None:
    raise OSError(errno.ENOSYS, "renameat2 is unavailable")
  renameat2.argtypes = (
      ctypes.c_int,
      ctypes.c_char_p,
      ctypes.c_int,
      ctypes.c_char_p,
      ctypes.c_uint,
  )
  renameat2.restype = ctypes.c_int
  result = renameat2(
      source_directory_fd,
      os.fsencode(source_name),
      destination_directory_fd,
      os.fsencode(destination_name),
      _RENAME_NOREPLACE,
  )
  if result != 0:
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error), destination_name)


def _write_all(fd: int, data: bytes) -> None:
  offset = 0
  while offset < len(data):
    written = os.write(fd, data[offset:])
    if written <= 0:
      raise OSError("short write while publishing autotuning cache")
    offset += written


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
  return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _unlink_owned_name(directory_fd: int, name: str, owned: os.stat_result) -> None:
  try:
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
  except FileNotFoundError:
    return
  if _same_inode(current, owned):
    os.unlink(name, dir_fd=directory_fd)
    os.fsync(directory_fd)


def build_candidate(
    result: api.AutotuningResult,
    identity: ExecutionIdentity,
    lowered: jax.stages.Lowered,
    output_directory_fd: int,
    output_name: str,
) -> PublishedArtifact:
  """Builds one durable no-clobber registrar-inbox candidate."""
  if not isinstance(identity, ExecutionIdentity):
    raise TypeError("identity must be an ExecutionIdentity")
  _verify_live_environment(identity)
  if not isinstance(output_name, str) or _OUTPUT_NAME_RE.fullmatch(
      output_name
  ) is None:
    raise ValueError("output_name must be one bounded ASCII .json component")
  directory_fd = fcntl.fcntl(
      output_directory_fd, fcntl.F_DUPFD_CLOEXEC, 0
  )
  temporary_name = f".tokamax-cache.{os.getpid()}.{secrets.token_hex(16)}.tmp"
  temporary_stat = None
  temporary_fd = None
  completed = False
  try:
    if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
      raise ValueError("output_directory_fd is not a directory")
    data = _artifact_bytes(result, identity, lowered)
    if len(data) > _MAX_ARTIFACT_BYTES:
      raise ValueError("autotuning cache artifact exceeds the size limit")
    temporary_fd = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o400,
        dir_fd=directory_fd,
    )
    temporary_stat = os.fstat(temporary_fd)
    os.fchmod(temporary_fd, 0o400)
    _write_all(temporary_fd, data)
    os.fsync(temporary_fd)
    temporary_stat = os.fstat(temporary_fd)
    if not stat.S_ISREG(temporary_stat.st_mode):
      raise RuntimeError("published cache candidate is not a regular file")
    if stat.S_IMODE(temporary_stat.st_mode) != 0o400:
      raise RuntimeError("published cache candidate mode is not 0400")
    if temporary_stat.st_nlink != 1 or temporary_stat.st_size != len(data):
      raise RuntimeError("published cache candidate inode is invalid")
    os.fsync(directory_fd)
    _rename_noreplace(directory_fd, temporary_name, directory_fd, output_name)
    os.fsync(directory_fd)
    readback_fd = os.open(
        output_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory_fd
    )
    try:
      readback_stat = os.fstat(readback_fd)
      if not _same_inode(temporary_stat, readback_stat):
        raise RuntimeError("published cache candidate inode changed")
      readback = _pread_exact(readback_fd, len(data))
      if readback != data:
        raise RuntimeError("published cache candidate readback mismatch")
      if _observed_stat(os.fstat(readback_fd)) != _observed_stat(readback_stat):
        raise RuntimeError("published cache candidate changed during readback")
    finally:
      os.close(readback_fd)
    completed = True
    return PublishedArtifact(len(data), _sha256(data), identity.sha256)
  finally:
    if temporary_fd is not None:
      os.close(temporary_fd)
    if temporary_stat is not None:
      _unlink_owned_name(directory_fd, temporary_name, temporary_stat)
      if not completed:
        _unlink_owned_name(directory_fd, output_name, temporary_stat)
    os.close(directory_fd)


def _pread_exact(fd: int, size: int) -> bytes:
  chunks = []
  offset = 0
  while offset < size:
    chunk = os.pread(fd, min(1024 * 1024, size - offset), offset)
    if not chunk:
      raise ValueError("autotuning cache artifact ended early")
    chunks.append(chunk)
    offset += len(chunk)
  if os.pread(fd, 1, size):
    raise ValueError("autotuning cache artifact exceeds its declared size")
  return b"".join(chunks)


def _capability_stat(capability: RegistrarReadCapability) -> tuple[Any, ...]:
  values = (
      capability.target_device,
      capability.target_inode,
      capability.target_uid,
      capability.target_gid,
      capability.target_mode,
      capability.target_nlink,
      capability.target_size,
  )
  if any(type(value) is not int or value < 0 for value in values):  # pylint: disable=unidiomatic-typecheck
    raise ValueError("registrar capability stat fields are invalid")
  return values


def _observed_stat(value: os.stat_result) -> tuple[int, ...]:
  return (
      value.st_dev,
      value.st_ino,
      value.st_uid,
      value.st_gid,
      stat.S_IMODE(value.st_mode),
      value.st_nlink,
      value.st_size,
  )


def _read_capability(capability: RegistrarReadCapability) -> bytes:
  if capability.artifact_kind != ARTIFACT_KIND:
    raise ValueError("registrar capability has the wrong artifact kind")
  if capability.kind_metadata_schema != KIND_METADATA_SCHEMA:
    raise ValueError("registrar capability has the wrong kind-metadata schema")
  _require_digest("target_sha256", capability.target_sha256)
  _require_digest("identity_sha256", capability.identity_sha256)
  _require_digest("receipt_sha256", capability.receipt_sha256)
  expected_stat = _capability_stat(capability)
  if capability.target_mode != 0o400 or capability.target_nlink != 1:
    raise ValueError("registrar capability does not describe a sealed file")
  if capability.target_size <= 0 or capability.target_size > _MAX_ARTIFACT_BYTES:
    raise ValueError("registrar capability size is invalid")
  if (  # pylint: disable=unidiomatic-typecheck
      type(capability.target_fd) is not int or capability.target_fd < 0
  ):
    raise ValueError("registrar capability fd is invalid")
  if not fcntl.fcntl(capability.target_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC:
    raise ValueError("registrar capability fd is not close-on-exec")
  fd = fcntl.fcntl(capability.target_fd, fcntl.F_DUPFD_CLOEXEC, 0)
  try:
    descriptor_flags = fcntl.fcntl(fd, fcntl.F_GETFD)
    status_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    if not descriptor_flags & fcntl.FD_CLOEXEC:
      raise ValueError("registrar capability fd is not close-on-exec")
    if status_flags & os.O_ACCMODE != os.O_RDONLY:
      raise ValueError("registrar capability fd is not read-only")
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or _observed_stat(before) != expected_stat:
      raise ValueError("registrar capability fd stat does not match its attestation")
    data = _pread_exact(fd, capability.target_size)
    after = os.fstat(fd)
    if _observed_stat(after) != expected_stat:
      raise ValueError("registrar capability fd changed while being read")
    if _sha256(data) != capability.target_sha256:
      raise ValueError("registrar capability content digest mismatch")
    return data
  finally:
    os.close(fd)


class RequiredAutotuningCache(contextlib.AbstractContextManager):
  """Exclusive cache overlay that rejects every cache miss."""

  def __init__(
      self, result: api.AutotuningResult, identity: ExecutionIdentity
  ):
    self._result = result
    self._identity = identity
    self._overlay = None
    self._context = None
    self._verified = False

  def __enter__(self) -> Self:
    if self._overlay is not None:
      raise RuntimeError("required autotuning cache context is already active")
    overlay = {}
    for bound_args, data in self._result.data:
      key = bound_args.autotuning_cache_key
      overlay.setdefault(bound_args.op, {}).setdefault(
          self._result.device_kind, {}
      )[key] = data
    overlay = op_lib._RequiredAutotuningCacheOverlay(overlay)  # pylint: disable=protected-access
    state = op_lib.get_autotuning_cache_overlay_state()
    if any(
        isinstance(item, op_lib._RequiredAutotuningCacheOverlay)  # pylint: disable=protected-access
        for item in state.stack
    ):
      raise RuntimeError("another required autotuning cache context is active")
    state.stack.append(overlay)
    context = state.context(state.context.value + (id(self),))
    try:
      context.__enter__()
    except BaseException:
      state.stack.pop()
      raise
    self._overlay = overlay
    self._context = context
    return self

  def verify_lowered(self, lowered: jax.stages.Lowered) -> None:
    if self._overlay is None:
      raise RuntimeError("required autotuning cache context is not active")
    if lowered_program_sha256(lowered) != self._identity.lowered_program_sha256:
      raise ValueError("live lowered program does not match the cache identity")
    self._verified = True

  def __exit__(self, exc_type, exc_value, traceback):
    if self._overlay is None or self._context is None:
      raise RuntimeError("required autotuning cache context is not active")
    state = op_lib.get_autotuning_cache_overlay_state()
    if not state.stack or state.stack[-1] is not self._overlay:
      raise RuntimeError("required autotuning cache stack is corrupted")
    verified = self._verified
    try:
      self._context.__exit__(exc_type, exc_value, traceback)
    finally:
      state.stack.pop()
      self._overlay = None
      self._context = None
      self._verified = False
    if exc_type is None and not verified:
      raise RuntimeError("live lowered program was not verified")


def load_required(
    capability: RegistrarReadCapability,
    expected_identity: ExecutionIdentity,
    expected_receipt_sha256: str,
) -> RequiredAutotuningCache:
  """Loads one exact cache from a registrar-authorized held file capability."""
  if not isinstance(expected_identity, ExecutionIdentity):
    raise TypeError("expected_identity must be an ExecutionIdentity")
  _require_digest("expected_receipt_sha256", expected_receipt_sha256)
  if capability.receipt_sha256 != expected_receipt_sha256:
    raise ValueError("registrar capability has the wrong artifact receipt")
  _verify_live_environment(expected_identity)
  data = _read_capability(capability)
  value = _strict_json(data)
  if _canonical_json(value) != data:
    raise ValueError("autotuning cache artifact encoding is not canonical")
  if type(value) is not dict:  # pylint: disable=unidiomatic-typecheck
    raise ValueError("autotuning cache artifact must be an object")
  expected_fields = {
      "schema",
      "identity",
      "identity_sha256",
      "payload_schema",
      "payload_sha256",
      "payload",
  }
  if set(value) != expected_fields:
    raise ValueError("autotuning cache artifact has missing or unknown fields")
  if value["schema"] != ARTIFACT_KIND or value["payload_schema"] != _PAYLOAD_SCHEMA:
    raise ValueError("autotuning cache artifact schema is unsupported")
  identity = ExecutionIdentity.from_dict(value["identity"])
  if identity != expected_identity:
    raise ValueError("autotuning cache execution identity mismatch")
  if value["identity_sha256"] != identity.sha256:
    raise ValueError("autotuning cache identity digest mismatch")
  if capability.identity_sha256 != identity.sha256:
    raise ValueError("registrar kind metadata does not match cache identity")
  if not isinstance(value["payload"], str):
    raise TypeError("autotuning cache payload must be text")
  payload = value["payload"]
  if value["payload_sha256"] != _sha256(payload.encode("utf-8")):
    raise ValueError("autotuning cache payload digest mismatch")
  _strict_json(payload)
  result = api.AutotuningResult.loads(payload)
  if _canonical_result(result) != payload:
    raise ValueError("autotuning cache payload encoding is not canonical")
  if result.device_kind != expected_identity.device_kind:
    raise ValueError("autotuning result device kind mismatch")
  return RequiredAutotuningCache(result, identity)
