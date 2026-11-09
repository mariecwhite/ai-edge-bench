"""Native macOS process, CPU and GPU probes (ctypes, standard library only).

Used by aeb.monitor when the harness runs natively on macOS, where there is no
/proc or cgroup. Everything here works without root:

  * proc_pid_rusage(RUSAGE_INFO_V6): CPU time, resident size, physical
    footprint (private memory, including Metal/IOAccelerator allocations made
    by the process), lifetime-maximum footprint and the kernel's per-process
    energy estimate.
  * proc_pidinfo(PROC_PIDTASKINFO): thread count.
  * host_statistics(HOST_CPU_LOAD_INFO): whole-machine busy/total CPU ticks.
  * IOKit IOAccelerator "PerformanceStatistics": system-wide GPU utilization
    and GPU-mapped system memory (includes the window server and any other
    GPU client; it is not per process).
  * IOKit AGXDeviceUserClient "AppUsage": accumulated GPU time of every Metal
    client, attributed to the process that created it (IOUserClientCreator).

CPU times in rusage are Mach absolute-time units on Apple silicon; they are
converted with mach_timebase_info.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import sys

IS_DARWIN = sys.platform == "darwin"

_u64 = ctypes.c_uint64


class _RusageInfoV6(ctypes.Structure):
  _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, _u64) for n in (
      "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
      "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
      "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
      "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
      "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
      "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
      "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
      "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
      "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
      "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy",
      "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time", "ri_flags",
      "ri_user_ptime", "ri_system_ptime", "ri_pinstructions", "ri_pcycles", "ri_energy_nj",
      "ri_penergy_nj", "ri_secure_time_in_system", "ri_secure_ptime_in_system",
      "ri_neural_footprint", "ri_lifetime_max_neural_footprint",
      "ri_interval_max_neural_footprint", "ri_conclave_footprint", "ri_page_wait_time_mach",
      "ri_page_cache_hits")] + [("ri_reserved", _u64 * 6)]


class _ProcTaskInfo(ctypes.Structure):
  _fields_ = [(n, _u64) for n in ("pti_virtual_size", "pti_resident_size", "pti_total_user",
                                   "pti_total_system", "pti_threads_user",
                                   "pti_threads_system")] + \
             [(n, ctypes.c_int32) for n in ("pti_policy", "pti_faults", "pti_pageins",
                                             "pti_cow_faults", "pti_messages_sent",
                                             "pti_messages_received", "pti_syscalls_mach",
                                             "pti_syscalls_unix", "pti_csw", "pti_threadnum",
                                             "pti_numrunning", "pti_priority")]


class _Timebase(ctypes.Structure):
  _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


RUSAGE_INFO_V6 = 6
PROC_PIDTASKINFO = 4
HOST_CPU_LOAD_INFO = 3
CPU_STATE_IDLE = 2
_kCFNumberSInt64Type = 4
_kCFStringEncodingUTF8 = 0x08000100

if IS_DARWIN:
  _libc = ctypes.CDLL(ctypes.util.find_library("System"), use_errno=True)
  _libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
  _libc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, _u64, ctypes.c_void_p, ctypes.c_int]
  _libc.mach_host_self.restype = ctypes.c_uint32
  _libc.host_statistics.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_uint32)]
  _tb = _Timebase()
  _libc.mach_timebase_info(ctypes.byref(_tb))
  _TB_NUMER, _TB_DENOM = _tb.numer, _tb.denom
  _HOST = _libc.mach_host_self()

  _cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
  _iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))
  _vp = ctypes.c_void_p
  _cf.CFStringCreateWithCString.restype = _vp
  _cf.CFStringCreateWithCString.argtypes = [_vp, ctypes.c_char_p, ctypes.c_uint32]
  _cf.CFDictionaryGetValue.restype = _vp
  _cf.CFDictionaryGetValue.argtypes = [_vp, _vp]
  _cf.CFNumberGetValue.restype = ctypes.c_bool
  _cf.CFNumberGetValue.argtypes = [_vp, ctypes.c_int, _vp]
  _cf.CFArrayGetCount.restype = ctypes.c_long
  _cf.CFArrayGetCount.argtypes = [_vp]
  _cf.CFArrayGetValueAtIndex.restype = _vp
  _cf.CFArrayGetValueAtIndex.argtypes = [_vp, ctypes.c_long]
  _cf.CFStringGetCString.restype = ctypes.c_bool
  _cf.CFStringGetCString.argtypes = [_vp, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
  _cf.CFRelease.argtypes = [_vp]
  _iokit.IOServiceMatching.restype = _vp
  _iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
  _iokit.IOServiceGetMatchingServices.argtypes = [ctypes.c_uint32, _vp,
                                                  ctypes.POINTER(ctypes.c_uint32)]
  _iokit.IOIteratorNext.restype = ctypes.c_uint32
  _iokit.IOIteratorNext.argtypes = [ctypes.c_uint32]
  _iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]
  _iokit.IORegistryEntryCreateCFProperty.restype = _vp
  _iokit.IORegistryEntryCreateCFProperty.argtypes = [ctypes.c_uint32, _vp, _vp, ctypes.c_uint32]
  _iokit.IORegistryEntryGetChildIterator.argtypes = [ctypes.c_uint32, ctypes.c_char_p,
                                                     ctypes.POINTER(ctypes.c_uint32)]

  def _cfstr(s: str):
    return _cf.CFStringCreateWithCString(None, s.encode(), _kCFStringEncodingUTF8)

  _K = {k: _cfstr(k) for k in ("PerformanceStatistics", "Device Utilization %",
                               "In use system memory", "Alloc system memory",
                               "IOUserClientCreator", "AppUsage", "accumulatedGPUTime")}


def mach_to_ns(t: int) -> int:
  return t * _TB_NUMER // _TB_DENOM


def proc_rusage(pid: int) -> dict | None:
  """CPU time (ns), memory (bytes) and energy (nJ) of a live process."""
  ri = _RusageInfoV6()
  if _libc.proc_pid_rusage(pid, RUSAGE_INFO_V6, ctypes.byref(ri)) != 0:
    return None
  return {
      "cpu_ns": mach_to_ns(ri.ri_user_time + ri.ri_system_time),
      "resident_bytes": ri.ri_resident_size,
      "footprint_bytes": ri.ri_phys_footprint,
      "footprint_max_bytes": ri.ri_lifetime_max_phys_footprint,
      "energy_nj": ri.ri_energy_nj,
  }


def proc_threads(pid: int) -> int | None:
  ti = _ProcTaskInfo()
  n = _libc.proc_pidinfo(pid, PROC_PIDTASKINFO, 0, ctypes.byref(ti), ctypes.sizeof(ti))
  return ti.pti_threadnum if n == ctypes.sizeof(ti) else None


def host_cpu_ticks() -> tuple[int, int]:
  """(busy, total) CPU ticks summed over all cores since boot."""
  buf = (ctypes.c_uint32 * 4)()
  count = ctypes.c_uint32(4)
  if _libc.host_statistics(_HOST, HOST_CPU_LOAD_INFO, buf, ctypes.byref(count)) != 0:
    return 0, 0
  total = sum(buf)
  return total - buf[CPU_STATE_IDLE], total


def _services(cls: bytes):
  it = ctypes.c_uint32(0)
  if _iokit.IOServiceGetMatchingServices(0, _iokit.IOServiceMatching(cls), ctypes.byref(it)) != 0:
    return
  yield from _drain(it)


def _children(entry: int):
  it = ctypes.c_uint32(0)
  if _iokit.IORegistryEntryGetChildIterator(entry, b"IOService", ctypes.byref(it)) != 0:
    return
  yield from _drain(it)


def _drain(it: ctypes.c_uint32):
  try:
    while True:
      obj = _iokit.IOIteratorNext(it.value)
      if not obj:
        break
      try:
        yield obj
      finally:
        _iokit.IOObjectRelease(obj)
  finally:
    _iokit.IOObjectRelease(it.value)


def _num(d, key) -> int | None:
  v = _cf.CFDictionaryGetValue(d, _K[key])
  if not v:
    return None
  out = ctypes.c_int64(0)
  return out.value if _cf.CFNumberGetValue(v, _kCFNumberSInt64Type, ctypes.byref(out)) else None


def gpu_device_stats() -> dict | None:
  """System-wide GPU utilization (%) and GPU-mapped system memory (bytes)."""
  for svc in _services(b"IOAccelerator"):
    d = _iokit.IORegistryEntryCreateCFProperty(svc, _K["PerformanceStatistics"], None, 0)
    if not d:
      continue
    try:
      return {"device_util_pct": _num(d, "Device Utilization %"),
              "in_use_mem_bytes": _num(d, "In use system memory"),
              "alloc_mem_bytes": _num(d, "Alloc system memory")}
    finally:
      _cf.CFRelease(d)
  return None


_CREATOR = re.compile(rb"pid (\d+)")


def gpu_time_ns(pid: int) -> int | None:
  """Accumulated GPU time of every Metal client created by `pid` (None if the
  process has not opened the GPU). User clients are registry children of the
  IOAccelerator service, not matchable services themselves."""
  total, found = 0, False
  buf = ctypes.create_string_buffer(256)
  for acc in _services(b"IOAccelerator"):
    for svc in _children(acc):
      s = _iokit.IORegistryEntryCreateCFProperty(svc, _K["IOUserClientCreator"], None, 0)
      if not s:
        continue
      try:
        ok = _cf.CFStringGetCString(s, buf, len(buf), _kCFStringEncodingUTF8)
      finally:
        _cf.CFRelease(s)
      m = _CREATOR.match(buf.value) if ok else None
      if not m or int(m.group(1)) != pid:
        continue
      found = True
      arr = _iokit.IORegistryEntryCreateCFProperty(svc, _K["AppUsage"], None, 0)
      if not arr:
        continue
      try:
        for i in range(_cf.CFArrayGetCount(arr)):
          total += _num(_cf.CFArrayGetValueAtIndex(arr, i), "accumulatedGPUTime") or 0
      finally:
        _cf.CFRelease(arr)
  return total if found else None
