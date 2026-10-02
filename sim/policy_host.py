import json
import os
import sys
import time
import traceback


def _to_jsonable(o):
    if hasattr(o, "item") and callable(o.item):
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist") and callable(o.tolist):
        return o.tolist()
    raise TypeError("not JSON serializable: %r" % type(o))


def _install_sandbox(policy_path, blocked):
    cwd = os.getcwd()
    roots = set()
    for p in list(sys.path) + [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix]:
        if not p:
            continue
        p = os.path.realpath(p)
        if p in (os.sep, cwd) or os.path.dirname(p) == p or not os.path.exists(p):
            continue
        roots.add(p.rstrip(os.sep) + os.sep)
        roots.add(p)
    roots = tuple(roots)
    policy_real = os.path.realpath(policy_path)
    policy_dir = os.path.dirname(policy_real)

    deny_events = {
        "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
        "os.fork", "os.forkpty", "os.kill", "os.killpg", "pty.spawn",
        "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname",
        "socket.gethostbyaddr", "socket.sendto", "urllib.Request",
        "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown",
        "os.truncate", "os.symlink", "os.link", "os.chdir",
        "shutil.rmtree", "shutil.copyfile", "shutil.move", "shutil.make_archive",
        "webbrowser.open", "sqlite3.connect",
    }

    def allowed_path(path):
        try:
            p = os.path.realpath(os.fsdecode(path))
        except Exception:
            return False
        if p == policy_real:
            return True
        return p in roots or p.startswith(roots)

    def cache_probe(path):
        try:
            return os.path.realpath(os.fsdecode(path)).startswith(os.path.join(policy_dir, "__pycache__") + os.sep)
        except Exception:
            return False

    def hook(event, args):
        if event == "open":
            path, mode = args[0], args[1]
            if isinstance(path, int) or path is None:
                return
            writing = isinstance(mode, str) and any(c in mode for c in "wax+")
            if writing or (isinstance(mode, int) and mode & (os.O_WRONLY | os.O_RDWR)):
                blocked.append("write")
                raise PermissionError("sandbox: writing files is not allowed")
            if cache_probe(path):
                raise PermissionError("sandbox: no bytecode cache")
            if not allowed_path(path):
                blocked.append("open")
                raise PermissionError("sandbox: file access is not allowed")
        elif event in ("os.listdir", "os.scandir"):
            path = args[0] if args else "."
            if isinstance(path, int) or not allowed_path(path if path is not None else "."):
                blocked.append("listdir")
                raise PermissionError("sandbox: directory listing is not allowed")
        elif event in deny_events or event.startswith("os.exec") or event.startswith("os.spawn"):
            blocked.append(event)
            raise PermissionError("sandbox: %s is not allowed" % event)

    sys.addaudithook(hook)


def main():
    policy_path = sys.argv[1]
    sys.dont_write_bytecode = True
    sys.path = [p for p in sys.path if p not in ("", os.getcwd())]

    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    inp = sys.stdin

    blocked = []
    reported = 0
    _install_sandbox(policy_path, blocked)

    def send(msg):
        nonlocal reported
        if len(blocked) > reported:
            msg["blocked"] = sorted(set(blocked[reported:]))
            reported = len(blocked)
        try:
            line = json.dumps(msg, default=_to_jsonable)
        except (TypeError, ValueError):
            line = json.dumps({"ok": True, "action": None, "note": "action not JSON serializable"})
        proto.write(line + "\n")
        proto.flush()

    policy = None
    for line in inp:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        cmd = msg.get("cmd")
        if cmd == "init":
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("candidate_policy", policy_path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules["candidate_policy"] = mod
                spec.loader.exec_module(mod)
                if not hasattr(mod, "Policy"):
                    send({"ok": False, "kind": "import", "error": "no class named Policy in submission"})
                    continue
                policy = mod.Policy(msg["info"])
                send({"ok": True})
            except BaseException as e:
                send({"ok": False, "kind": "init", "error": "%s: %s" % (type(e).__name__, e),
                      "traceback": traceback.format_exc()[-4000:]})
        elif cmd == "act":
            try:
                action = policy.act(msg["obs"])
                send({"ok": True, "action": action})
            except BaseException as e:
                send({"ok": False, "kind": "act", "error": "%s: %s" % (type(e).__name__, e),
                      "traceback": traceback.format_exc()[-4000:]})
        elif cmd == "close":
            break


if __name__ == "__main__":
    main()
