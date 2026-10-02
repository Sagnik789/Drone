import json
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from .env import Episode

_HOST_SRC = pathlib.Path(__file__).with_name("policy_host.py").read_text()
_MEM_LIMIT_BYTES = 4 * 1024 ** 3


class PolicyTimeout(Exception):
    pass


class PolicyDied(Exception):
    pass


def _child_limits():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def _child_env(home):
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": home,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    here = str(pathlib.Path(__file__).resolve().parents[1])
    pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
          if p and not os.path.realpath(p).startswith(here)]
    if pp:
        env["PYTHONPATH"] = os.pathsep.join(pp)
    for k in ("VIRTUAL_ENV", "CONDA_PREFIX", "LD_LIBRARY_PATH"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


class SubprocessPolicy:

    def __init__(self, policy_path, show_output=False):
        self.policy_path = os.path.abspath(policy_path)
        self.show_output = show_output
        self.proc = None
        self.tmpdir = None
        self._lines = queue.Queue()
        self.blocked = set()

    def start(self, info, timeout):
        self.tmpdir = tempfile.mkdtemp(prefix="policy_run_")
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _HOST_SRC, self.policy_path],
            cwd=self.tmpdir, env=_child_env(self.tmpdir),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None if self.show_output else subprocess.DEVNULL,
            **({} if os.name == "nt" else {"preexec_fn": _child_limits, "start_new_session": True}))
        threading.Thread(target=self._reader, daemon=True).start()
        return self._call({"cmd": "init", "info": info}, timeout)

    def _reader(self):
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def act(self, obs, timeout):
        return self._call({"cmd": "act", "obs": obs}, timeout)

    def _call(self, msg, timeout):
        data = (json.dumps(msg) + "\n").encode()
        t0 = time.perf_counter()
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise PolicyDied("policy process exited (code %s)" % self.proc.poll())
        line = self._readline(t0 + timeout)
        elapsed = time.perf_counter() - t0
        try:
            reply = json.loads(line)
        except ValueError:
            raise PolicyDied("garbled reply from policy process")
        for b in reply.get("blocked", []):
            self.blocked.add(b)
        return reply, elapsed

    def _readline(self, deadline):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise PolicyTimeout()
        try:
            line = self._lines.get(timeout=remaining)
        except queue.Empty:
            raise PolicyTimeout()
        if line is None:
            raise PolicyDied("policy process exited (code %s)" % self.proc.wait())
        return line

    def close(self):
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    try:
                        self.proc.stdin.write(b'{"cmd": "close"}\n')
                        self.proc.stdin.flush()
                        self.proc.wait(timeout=0.2)
                    except Exception:
                        pass
                if self.proc.poll() is None:
                    self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except Exception:
                    pass
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)


class InProcessPolicy:

    def __init__(self, policy_path, show_output=True):
        self.policy_path = os.path.abspath(policy_path)
        self.policy = None
        self.blocked = set()

    def start(self, info, timeout):
        import importlib.util
        t0 = time.perf_counter()
        try:
            name = "policy_%d" % abs(hash(self.policy_path))
            spec = importlib.util.spec_from_file_location(name, self.policy_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            if not hasattr(mod, "Policy"):
                return {"ok": False, "kind": "import", "error": "no class named Policy in submission"}, 0.0
            self.policy = mod.Policy(json.loads(json.dumps(info)))
        except Exception as e:
            return {"ok": False, "kind": "init", "error": "%s: %s" % (type(e).__name__, e)}, 0.0
        elapsed = time.perf_counter() - t0
        if elapsed > timeout:
            raise PolicyTimeout()
        return {"ok": True}, elapsed

    def act(self, obs, timeout):
        t0 = time.perf_counter()
        try:
            action = self.policy.act(json.loads(json.dumps(obs)))
        except Exception as e:
            return {"ok": False, "kind": "act", "error": "%s: %s" % (type(e).__name__, e)}, 0.0
        elapsed = time.perf_counter() - t0
        if elapsed > timeout:
            raise PolicyTimeout()
        return {"ok": True, "action": action}, elapsed

    def close(self):
        self.policy = None


def run_episode(policy_path, scenario, isolate=True, record_trace=False, show_output=False,
                config=None, keep_episode=False):
    ep = Episode(scenario, config=config, record_trace=record_trace)
    lim = ep.limits
    act_limit = lim["act_time_limit_s"] + lim["ipc_grace_s"]
    total_limit = lim["episode_decision_time_limit_s"]
    pol = SubprocessPolicy(policy_path, show_output) if isolate else InProcessPolicy(policy_path)

    times = []
    total = 0.0
    timeout = crash = False
    error = ""
    trace = ""
    try:
        try:
            reply, _ = pol.start(ep.info(), lim["init_time_limit_s"])
            if not reply.get("ok"):
                crash, error = True, reply.get("error", "init failed")
                trace = reply.get("traceback", "")
                ep.end("crash")
        except PolicyTimeout:
            timeout, error = True, "init exceeded %.1f s" % lim["init_time_limit_s"]
            ep.end("timeout")
        except PolicyDied as e:
            crash, error = True, str(e)
            ep.end("crash")

        while not ep.done:
            obs = ep.observe()
            budget_left = total_limit - total
            limit = min(act_limit, budget_left + lim["ipc_grace_s"])
            try:
                reply, elapsed = pol.act(obs, limit)
            except PolicyTimeout:
                timeout = True
                if limit < act_limit:
                    error = "total decision time exceeded %.0f s" % total_limit
                    ep.end("episode_time_limit")
                else:
                    error = "act() exceeded %.0f ms at action %d" % (1000 * lim["act_time_limit_s"], ep.n_actions)
                    ep.end("timeout")
                break
            except PolicyDied as e:
                crash, error = True, str(e)
                ep.end("crash")
                break
            times.append(elapsed)
            total += elapsed
            if not reply.get("ok"):
                crash, error = True, reply.get("error", "act failed")
                trace = reply.get("traceback", "")
                ep.end("crash")
                break
            ep.step(reply.get("action"))
    finally:
        pol.close()

    res = ep.result()
    res.update({
        "timeout": timeout,
        "crash": crash,
        "error": error,
        "traceback": trace,
        "n_calls": len(times),
        "decision_ms_mean": 1000.0 * (sum(times) / len(times)) if times else 0.0,
        "decision_ms_max": 1000.0 * max(times) if times else 0.0,
        "decision_s_total": total,
        "blocked": sorted(pol.blocked),
    })
    if record_trace:
        res["trace"] = ep.trace_array()
        res["detections_log"] = ep.all_detections
    if keep_episode:
        res["episode"] = ep
    return res
