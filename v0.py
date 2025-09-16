
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Samsoft DOS 1.0x — a playful, MS‑DOS–style shell in Python with a built‑in
`gssplat` command that runs a minimal Gaussian‑splatting projection scaffold.

• Works on modern Windows 11 (and other OSes with Python 3.9+).
• Sandboxed: all file commands operate within a chosen root (default: CWD).
• `gssplat` lazily imports torch/pycolmap/matplotlib when invoked, and
  implements the user's earlier scaffold (adapted to avoid missing helpers).

USAGE
-----
    python SamsoftDos1.0x.py                 # start interactive shell at CWD
    python SamsoftDos1.0x.py --root D:\work  # start at a chosen root
    python SamsoftDos1.0x.py --cmd "dir"     # run a single command and exit

Inside the shell:
    C:\>help
    C:\>dir [path]
    C:\>cd [path]
    C:\>mkdir <name>      C:\>rmdir <name> [/s]
    C:\>copy <src> <dst>  C:\>move <src> <dst>  C:\>ren <old> <new>
    C:\>type <file>       C:\>del <pattern>
    C:\>ver               C:\>cls              C:\>exit

    C:\>gssplat --colmap <COLMAP_recon_dir> [--image-id 100] [--plot]
        Loads points/cameras from a COLMAP sparse reconstruction, projects
        points into the selected image, and (optionally) shows a scatter plot.
        This is a stepping stone toward full differentiable Gaussian splatting.
        Dependencies are imported on-demand:
            pip install pycolmap torch numpy matplotlib

NOTES
-----
• This is *not* a full OS nor an emulator. It’s a safe, toy shell.
• Deletions/copies/moves are restricted to the sandbox root you choose.
"""

import argparse
import os
import shlex
import sys
import shutil
import glob
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath

SAMSOFT_NAME = "Samsoft DOS"
SAMSOFT_VERSION = "1.0x"
BANNER = rf"""{SAMSOFT_NAME} {SAMSOFT_VERSION}
(C) Samsoft Industries. All rights reserved.
Type HELP for available commands.
"""

# --------------------------- Utility helpers ----------------------------

def win_now_str(ts=None):
    dt = datetime.fromtimestamp(ts) if ts is not None else datetime.now()
    return dt.strftime("%m/%d/%Y  %I:%M %p")

def fmt_commas(n: int) -> str:
    return f"{n:,}"

def is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False

def force_within_root(p: Path, root: Path) -> Path:
    p = p.resolve()
    root = root.resolve()
    if not is_subpath(p, root):
        raise PermissionError(f"Refusing to access outside sandbox root: {p}")
    return p

def to_abs(path_str: str, cwd: Path, root: Path) -> Path:
    """
    Convert a DOS-like path to an absolute host path confined within `root`.
    Accepts forms like: foo\bar, \foo, C:\foo (C: is ignored), "." and "..".
    """
    s = path_str.strip().strip('"').strip("'")
    if not s:
        return cwd

    # Strip leading drive like C:
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        s = s[2:]
    # Normalize slashes
    s = s.replace("\\", "/")

    if s.startswith("/"):
        candidate = (root / s[1:]).resolve()
    else:
        candidate = (cwd / s).resolve()
    return force_within_root(candidate, root)

def list_dir_display(path: Path, root: Path) -> str:
    if not path.exists():
        return f"File Not Found\n"

    lines = []
    lines.append(f" Volume in drive C has no label.")
    lines.append(f" Directory of {PureWindowsPath(str(path.resolve()).replace(str(root.resolve()), 'C:\\')).as_posix().replace('/', '\\')}")
    lines.append("")
    items = list(path.iterdir())
    dirs = [p for p in items if p.is_dir()]
    files = [p for p in items if p.is_file()]
    entries = sorted(dirs, key=lambda p: p.name.lower()) + sorted(files, key=lambda p: p.name.lower())

    file_count = 0
    file_bytes = 0
    for p in entries:
        st = p.stat()
        dt = win_now_str(st.st_mtime)
        name = p.name
        if p.is_dir():
            lines.append(f"{dt}    <DIR>          {name}")
        else:
            size = st.st_size
            file_count += 1
            file_bytes += size
            lines.append(f"{dt}           {fmt_commas(size):>12} {name}")
    lines.append(f"              {file_count} File(s)    {fmt_commas(file_bytes)} bytes")
    # Report free space for the underlying filesystem where root lives
    total, used, free = shutil.disk_usage(root)
    lines.append(f"              {len(dirs)} Dir(s)     {fmt_commas(free)} bytes free")
    return "\n".join(lines) + "\n"

def print_error(msg: str):
    print(f"Error: {msg}")

# --------------------------- Command handlers ---------------------------

@dataclass
class Context:
    root: Path   # sandbox root
    cwd: Path    # current directory (absolute path within root)

class SamsoftShell:
    def __init__(self, root: Path):
        root = root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"Root does not exist: {root}")
        self.ctx = Context(root=root, cwd=root)
        self.running = True
        self.commands = {
            "HELP": self.cmd_help,
            "?": self.cmd_help,
            "DIR": self.cmd_dir,
            "CD": self.cmd_cd,
            "CHDIR": self.cmd_cd,
            "MKDIR": self.cmd_mkdir,
            "MD": self.cmd_mkdir,
            "RMDIR": self.cmd_rmdir,
            "RD": self.cmd_rmdir,
            "COPY": self.cmd_copy,
            "MOVE": self.cmd_move,
            "DEL": self.cmd_del,
            "ERASE": self.cmd_del,
            "REN": self.cmd_ren,
            "RENAME": self.cmd_ren,
            "TYPE": self.cmd_type,
            "CLS": self.cmd_cls,
            "VER": self.cmd_ver,
            "DATE": self.cmd_date,
            "TIME": self.cmd_time,
            "ECHO": self.cmd_echo,
            "EXIT": self.cmd_exit,
            # Custom command:
            "GSSPLAT": self.cmd_gssplat,
        }

    # ---- Prompt ----
    def prompt(self) -> str:
        rel = str(self.ctx.cwd.resolve()).replace(str(self.ctx.root.resolve()), "C:\\")
        rel = rel.replace("/", "\\")
        if not rel.endswith("\\"):
            rel += "\\"
        return f"{rel}>"

    # ---- Built-ins ----
    def cmd_help(self, argv):
        if len(argv) > 1 and argv[1].upper() == "GSSPLAT":
            print(dedent("""\
                GSSPLAT — Minimal Gaussian-splatting projection scaffold
                Usage:
                  gssplat --colmap <path_to_COLMAP_sparse_reconstruction>
                          [--image-id 100] [--plot] [--max-points N]

                Description:
                  Loads a COLMAP reconstruction, extracts 3D points and camera
                  intrinsics/extrinsics for the chosen image, projects points
                  to image plane, and optionally shows a scatter plot.
                  This is a light educational scaffold; extend with rasterization
                  to produce full differentiable Gaussian splatting.

                Dependencies (imported only when you run this command):
                  pip install pycolmap torch numpy matplotlib
            """))
            return

        print(dedent(f"""\
            For more information on a specific command, type HELP command-name

            DIR       Displays a list of files and subdirectories.
            CD        Displays the name of or changes the current directory.
            MKDIR     Creates a directory.
            RMDIR     Removes a directory. Use RMDIR /S for recursive.
            COPY      Copies one or more files.
            MOVE      Moves one or more files.
            REN       Renames a file or files.
            DEL       Deletes one or more files (supports wildcards).
            TYPE      Displays the contents of a text file.
            CLS       Clears the screen.
            VER       Displays the Samsoft DOS version.
            DATE      Displays today's date.
            TIME      Displays the current time.
            ECHO      Displays messages.
            EXIT      Quits Samsoft DOS.

            GSSPLAT   Minimal Gaussian-splatting projection scaffold. Type HELP GSSPLAT.
        """))

    def cmd_dir(self, argv):
        path = self.ctx.cwd if len(argv) == 1 else to_abs(argv[1], self.ctx.cwd, self.ctx.root)
        if not path.exists():
            print("File Not Found")
            return
        print(list_dir_display(path, self.ctx.root), end="")

    def cmd_cd(self, argv):
        if len(argv) == 1:
            print(str(PureWindowsPath(str(self.ctx.cwd).replace(str(self.ctx.root), "C:\\"))))
            return
        dest = to_abs(argv[1], self.ctx.cwd, self.ctx.root)
        if not dest.exists() or not dest.is_dir():
            print("The system cannot find the path specified.")
            return
        self.ctx.cwd = dest

    def cmd_mkdir(self, argv):
        if len(argv) < 2:
            print("Usage: MKDIR <dir>")
            return
        dest = to_abs(argv[1], self.ctx.cwd, self.ctx.root)
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print_error(str(e))

    def cmd_rmdir(self, argv):
        if len(argv) < 2:
            print("Usage: RMDIR <dir> [/S]")
            return
        recursive = any(arg.upper() == "/S" for arg in argv[2:])
        dest = to_abs(argv[1], self.ctx.cwd, self.ctx.root)
        try:
            if recursive:
                shutil.rmtree(dest)
            else:
                dest.rmdir()
        except Exception as e:
            print_error(str(e))

    def cmd_copy(self, argv):
        if len(argv) < 3:
            print("Usage: COPY <src> <dst>")
            return
        srcs = []
        for s in argv[1:-1]:
            # Expand wildcards relative to cwd
            abs_pattern = to_abs(s, self.ctx.cwd, self.ctx.root)
            # Glob on the filesystem
            srcs.extend([p for p in abs_pattern.parent.glob(abs_pattern.name) if p.is_file()])
        dst = to_abs(argv[-1], self.ctx.cwd, self.ctx.root)
        if dst.exists() and dst.is_dir():
            for s in srcs:
                shutil.copy2(s, dst / s.name)
        elif len(srcs) == 1:
            shutil.copy2(srcs[0], dst)
        else:
            print("COPY: Destination must be a directory when copying multiple files.")

    def cmd_move(self, argv):
        if len(argv) < 3:
            print("Usage: MOVE <src> <dst>")
            return
        srcs = []
        for s in argv[1:-1]:
            abs_pattern = to_abs(s, self.ctx.cwd, self.ctx.root)
            srcs.extend([p for p in abs_pattern.parent.glob(abs_pattern.name)])
        dst = to_abs(argv[-1], self.ctx.cwd, self.ctx.root)
        if dst.exists() and dst.is_dir():
            for s in srcs:
                shutil.move(str(s), str(dst / s.name))
        elif len(srcs) == 1:
            shutil.move(str(srcs[0]), str(dst))
        else:
            print("MOVE: Destination must be a directory when moving multiple files.")

    def cmd_del(self, argv):
        if len(argv) < 2:
            print("Usage: DEL <pattern>")
            return
        deleted = 0
        for pattern in argv[1:]:
            abs_pattern = to_abs(pattern, self.ctx.cwd, self.ctx.root)
            for p in abs_pattern.parent.glob(abs_pattern.name):
                if p.is_file():
                    try:
                        p.unlink()
                        deleted += 1
                    except Exception as e:
                        print_error(str(e))
        print(f"Deleted {deleted} file(s).")

    def cmd_ren(self, argv):
        if len(argv) < 3:
            print("Usage: REN <old> <new>")
            return
        old = to_abs(argv[1], self.ctx.cwd, self.ctx.root)
        new = to_abs(argv[2], self.ctx.cwd, self.ctx.root)
        try:
            old.rename(new)
        except Exception as e:
            print_error(str(e))

    def cmd_type(self, argv):
        if len(argv) < 2:
            print("Usage: TYPE <file>")
            return
        p = to_abs(argv[1], self.ctx.cwd, self.ctx.root)
        if not p.exists() or not p.is_file():
            print("File Not Found")
            return
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                print(f.read(), end="")
        except Exception as e:
            print_error(str(e))

    def cmd_cls(self, argv):
        # Cross-platform clear without relying on external commands
        print("\033c", end="")

    def cmd_ver(self, argv):
        print(f"{SAMSOFT_NAME} Version {SAMSOFT_VERSION}")

    def cmd_date(self, argv):
        print(win_now_str()[:10])

    def cmd_time(self, argv):
        print(win_now_str()[12:])

    def cmd_echo(self, argv):
        print(" ".join(argv[1:]))

    def cmd_exit(self, argv):
        self.running = False

    # ----------------------- gssplat command ----------------------------
    def cmd_gssplat(self, argv):
        """
        Wraps a minimal Gaussian-splatting projection scaffold adapted from the
        user's snippet. Imports heavy deps only if invoked.
        """
        try:
            import math
            import numpy as np
            import torch
            import matplotlib.pyplot as plt
            import pycolmap
        except Exception as e:
            print_error(
                "Missing dependencies. Install with:\n"
                "  pip install pycolmap torch numpy matplotlib\n"
                f"Import error: {e}"
            )
            return

        # ---- Argparse for gssplat ----
        parser = argparse.ArgumentParser(prog="gssplat", add_help=False)
        parser.add_argument("--colmap", required=True, help="Path to COLMAP sparse reconstruction (e.g., sparse/0)")
        parser.add_argument("--image-id", type=int, default=100, help="Image ID to project into (default: 100)")
        parser.add_argument("--max-points", type=int, default=300000, help="Cap number of 3D points for speed")
        parser.add_argument("--plot", action="store_true", help="Show a 2D scatter of projected points")
        parser.add_argument("--headless", action="store_true", help="Skip showing plots")
        parser.add_argument("--quiet", action="store_true", help="Reduce console output")
        try:
            args = parser.parse_args(argv[1:])
        except SystemExit:
            # argparse already printed error/help
            return

        colmap_path = to_abs(args.colmap, self.ctx.cwd, self.ctx.root)
        if not colmap_path.exists():
            print_error(f"COLMAP path not found: {colmap_path}")
            return

        # ---- Helpers adapted from the user's snippet ----
        def qvec_to_rotmat(qvec: np.ndarray) -> torch.Tensor:
            """COLMAP uses [qw, qx, qy, qz] (scalar-first)."""
            qw, qx, qy, qz = qvec
            # Convert to torch for consistent downstream ops
            r00 = 1 - 2*(qy*qy + qz*qz)
            r01 = 2*(qx*qy - qz*qw)
            r02 = 2*(qx*qz + qy*qw)
            r10 = 2*(qx*qy + qz*qw)
            r11 = 1 - 2*(qx*qx + qz*qz)
            r12 = 2*(qy*qz - qx*qw)
            r20 = 2*(qx*qz - qy*qw)
            r21 = 2*(qy*qz + qx*qw)
            r22 = 1 - 2*(qx*qx + qy*qy)
            R = torch.tensor([[r00, r01, r02],
                              [r10, r11, r12],
                              [r20, r21, r22]], dtype=torch.float32)
            return R

        def get_extrinsic_matrix(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Homogeneous world->camera matrix [R|t] as 4x4."""
            Rt = torch.zeros((4, 4), dtype=torch.float32)
            Rt[:3, :3] = R
            Rt[:3, 3] = t
            Rt[3, 3] = 1.0
            return Rt

        def get_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
            """Homogeneous intrinsic matrix 4x4 (OpenGL-ish homogeneous form)."""
            return torch.tensor([
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=torch.float32)

        def project_points(points: torch.Tensor,
                           K: torch.Tensor,
                           Rt: torch.Tensor,
                           width: int,
                           height: int):
            """
            Project 3D world points to image plane.
            Returns (xy, depth, mask) with in-bounds visibility mask applied.
            """
            # Homogeneous
            ones = torch.ones((points.shape[0], 1), dtype=points.dtype)
            P_h = torch.cat([points, ones], dim=1)  # [N,4]

            # World -> camera
            C = (Rt @ P_h.T).T  # [N,4]
            # Camera -> image
            I = (K @ C.T).T      # [N,4]

            # Normalize
            z = I[:, 2].clone()
            # Prevent division by zero
            eps = 1e-8
            z = torch.where(z.abs() < eps, torch.full_like(z, eps), z)
            xy = I[:, :2] / z.unsqueeze(1)

            # Visibility / bounds
            mask = (z > 0) & \
                   (xy[:, 0] >= 0) & (xy[:, 0] < width) & \
                   (xy[:, 1] >= 0) & (xy[:, 1] < height)
            return xy[mask], z[mask], mask

        # ---- Load COLMAP reconstruction ----
        try:
            recon = pycolmap.Reconstruction(str(colmap_path))
        except Exception as e:
            print_error(f"Failed to load COLMAP reconstruction from {colmap_path}: {e}")
            return

        if not recon.points3D or not recon.images or not recon.cameras:
            print_error("COLMAP reconstruction missing points/images/cameras.")
            return

        # Points & colors
        points3d = []
        colors = []
        for p in recon.points3D.values():
            # Keep points seen in at least 2 images (like user's snippet)
            try:
                track_len = len(p.track.elements) if hasattr(p.track, "elements") else len(p.track)
            except Exception:
                # Fallback if pycolmap structure differs
                track_len = getattr(p.track, "length", lambda: 0)()
            if track_len >= 2:
                points3d.append(p.xyz)
                colors.append(p.color if hasattr(p, "color") else [255, 255, 255])
        if len(points3d) == 0:
            print_error("No 3D points found with track length >= 2.")
            return

        import numpy as _np
        pts_np = _np.asarray(points3d, dtype=_np.float32)
        if pts_np.shape[0] > args.max_points:
            pts_np = pts_np[_np.random.choice(pts_np.shape[0], args.max_points, replace=False)]

        pts = torch.from_numpy(pts_np)  # [N,3]

        # Pick image
        if args.image_id not in recon.images:
            # Choose the first available image id
            first_id = next(iter(recon.images.keys()))
            if not args.quiet:
                print(f"Image ID {args.image_id} not found; using {first_id} instead.")
            image = recon.images[first_id]
        else:
            image = recon.images[args.image_id]

        cam = recon.cameras[image.camera_id]

        # Rotation/translation from COLMAP pose (world->camera)
        # COLMAP's convention: x_cam = R * x_world + t
        qvec = getattr(image, "qvec", None)
        tvec = getattr(image, "tvec", None)
        if qvec is None or tvec is None:
            print_error("Image pose is missing qvec/tvec.")
            return
        R = qvec_to_rotmat(_np.asarray(qvec, dtype=_np.float32))
        t = torch.tensor(_np.asarray(tvec, dtype=_np.float32))

        Rt = get_extrinsic_matrix(R, t)

        # Camera intrinsics: support common models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, etc.
        fx = fy = cx = cy = None
        model_name = getattr(cam, "model_name", getattr(cam, "model", "PINHOLE"))
        params = _np.asarray(cam.params, dtype=_np.float32)
        width = int(getattr(cam, "width", 0))
        height = int(getattr(cam, "height", 0))

        mn = str(model_name).upper()
        try:
            if "PINHOLE" in mn and params.size >= 4:
                fx, fy, cx, cy = params[:4]
            elif "SIMPLE_PINHOLE" in mn and params.size >= 3:
                f, cx, cy = params[:3]
                fx = fy = f
            elif "OPENCV" in mn and params.size >= 4:
                fx, fy, cx, cy = params[:4]
            elif "SIMPLE_RADIAL" in mn and params.size >= 3:
                f, cx, cy = params[:3]
                fx = fy = f
            else:
                # Fallback: assume [fx, fy, cx, cy]
                if params.size >= 4:
                    fx, fy, cx, cy = params[:4]
                else:
                    raise ValueError(f"Unsupported camera model or insufficient params: {model_name} {params}")
        except Exception as e:
            print_error(str(e))
            return

        K = get_intrinsic_matrix(float(fx), float(fy), float(cx), float(cy))

        # Project
        xy, depth, mask = project_points(pts, K, Rt, width=width, height=height)
        if not args.quiet:
            print(f"Projected {xy.shape[0]} in-bounds points into image {getattr(image, 'name', 'unknown')} "
                  f"({width}x{height}).")

        # Optional scatter plot
        if args.plot and not args.headless:
            try:
                plt.figure()
                plt.scatter(xy[:, 0].cpu().numpy(), xy[:, 1].cpu().numpy(), s=0.1)
                plt.gca().invert_yaxis()
                plt.xlim(0, width)
                plt.ylim(height, 0)
                plt.title("Projected 3D points (Gaussian centers)")
                plt.xlabel("u")
                plt.ylabel("v")
                plt.show()
            except Exception as e:
                print_error(f"Plotting failed: {e}")

        print("Gaussian Splatting setup complete. Extend for full rendering.")

    # ----------------------------- Loop ---------------------------------
    def run(self, one_command: str | None = None):
        print(BANNER)
        if one_command:
            self.execute_line(one_command)
            return
        while self.running:
            try:
                line = input(self.prompt())
                self.execute_line(line)
            except (EOFError, KeyboardInterrupt):
                print("")
                break
            except PermissionError as e:
                print_error(str(e))
            except Exception as e:
                print_error(str(e))

    def execute_line(self, line: str):
        line = line.strip()
        if not line:
            return
        # Batch files: if user enters something like '@foo.bat'
        if line.startswith("@") and line.lower().endswith(".bat"):
            self._run_batch(line[1:])
            return

        # Tokenize (Windows-style quoting isn't perfect in shlex; good enough here)
        try:
            argv = shlex.split(line, posix=False)
        except Exception:
            argv = line.split()
        cmd = argv[0].upper()
        handler = self.commands.get(cmd)
        if handler is None:
            print(f"'{argv[0]}' is not recognized as an internal or external command.")
            return
        handler(argv)

    def _run_batch(self, batch_path_str: str):
        p = to_abs(batch_path_str, self.ctx.cwd, self.ctx.root)
        if not p.exists() or not p.is_file():
            print("File Not Found")
            return
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("::") or line.startswith("REM"):
                        continue
                    print(f"{self.prompt()}{line}")
                    self.execute_line(line)
        except Exception as e:
            print_error(str(e))

# ----------------------------- Entrypoint -------------------------------

def main():
    ap = argparse.ArgumentParser(description=f"{SAMSOFT_NAME} {SAMSOFT_VERSION}")
    ap.add_argument("--root", help="Sandbox root directory (default: current directory)")
    ap.add_argument("--cmd", help="Run a single command then exit")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else Path.cwd().resolve()

    shell = SamsoftShell(root=root)
    shell.run(one_command=args.cmd)

if __name__ == "__main__":
    main()
