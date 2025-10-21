import os
import argparse
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons, TextBox
from PIL import Image  

try:
    import tkinter as _tk
    from tkinter import filedialog as _fd
    _HAS_TK = True
except Exception:
    _HAS_TK = False

from .sokoban_solver import parse_level, solve_sokoban, State, DIRS
from .level_loader import load_boxoban_levels


def find_gym_sokoban_assets_dir() -> Optional[str]:
    try:
        import importlib.resources as ir
        import gym_sokoban
        for folder in ("sokoban_assets", "surface"):
            try:
                p = ir.files("gym_sokoban.envs") / folder
                if p and os.path.isdir(str(p)):
                    return str(p)
            except Exception:
                pass
        base = os.path.dirname(gym_sokoban.__file__)
        for folder in ("sokoban_assets", "surface"):
            cand = os.path.join(base, "envs", folder)
            if os.path.isdir(cand):
                return cand
    except Exception:
        pass
    return None

SPRITE_CANDIDATES = {
    "floor":       ["floor.png", "SokobanFloor.png"],
    "wall":        ["wall.png", "SokobanWall.png"],
    "goal":        ["box_target.png", "target.png", "Goal.png", "SokobanTarget.png"],
    "box":         ["box.png", "crate.png", "SokobanBox.png"],
    "box_goal":    ["box_on_target.png", "crate_on_target.png", "SokobanBoxOnTarget.png"],
    "player":      ["player.png", "SokobanPlayer.png", "player_1.png"],
    "player_goal": ["player_on_target.png", "SokobanPlayerOnTarget.png", "player_on_goal.png"],
}

def load_sprite(assets_dir: str, names: List[str], tile: int) -> Image.Image:
    last_err = None
    for name in names:
        path = os.path.join(assets_dir, name)
        if os.path.isfile(path):
            try:
                img = Image.open(path).convert("RGBA")
                if tile and img.size != (tile, tile):
                    img = img.resize((tile, tile), Image.NEAREST)  
                return img
            except Exception as e:
                last_err = e
    raise FileNotFoundError(f"Could not load any of {names} in {assets_dir}. Last error: {last_err}")

def load_sprites(assets_dir: str, tile: int) -> Dict[str, Image.Image]:
    return {k: load_sprite(assets_dir, SPRITE_CANDIDATES[k], tile) for k in SPRITE_CANDIDATES}

# ---------- rendering helpers ----------

def render_frame(level, state: State, sprites: Dict[str, Image.Image], tile: int, show_goals: bool) -> np.ndarray:
    H, W = level.height, level.width
    canvas = Image.new("RGBA", (W * tile, H * tile), (0, 0, 0, 0))

    floor = sprites["floor"]
    for i in range(H):
        for j in range(W):
            canvas.paste(floor, (j * tile, i * tile), floor)

    wall = sprites["wall"]
    for (i, j) in level.walls:
        canvas.paste(wall, (j * tile, i * tile), wall)

    if show_goals:
        goal = sprites["goal"]
        for (i, j) in level.goals:
            canvas.paste(goal, (j * tile, i * tile), goal)

    box = sprites["box"]
    box_goal = sprites["box_goal"]
    for (i, j) in state.boxes:
        spr = box_goal if (i, j) in level.goals and show_goals else box
        canvas.paste(spr, (j * tile, i * tile), spr)

    pi, pj = state.player
    player = sprites["player"]
    player_goal = sprites["player_goal"]
    spr = player_goal if ((pi, pj) in level.goals and show_goals) else player
    canvas.paste(spr, (pj * tile, pi * tile), spr)

    return np.array(canvas.convert("RGB"))

def step(state: State, mv: str) -> State:
    di, dj = DIRS[mv.upper()]
    ni, nj = state.player[0] + di, state.player[1] + dj
    boxes = set(state.boxes)
    if mv.isupper(): 
        bi, bj = ni + di, nj + dj
        boxes.remove((ni, nj))
        boxes.add((bi, bj))
    return State((ni, nj), frozenset(boxes))

def to_dirs(solution: str) -> List[str]:
    return [ch for ch in (solution or "") if ch.lower() in ("u", "d", "l", "r")]

# ---------- viewer with UI ----------

class Viewer:
    def __init__(self, level_pack_path: str, level_index: int, method: str,
                 tile: int, fps: float, assets_dir: str,
                 show_goals=True, show_grid=False, show_hud=True):
        self.level_pack_path = level_pack_path
        self.method = method
        self.tile = tile
        self.fps = fps
        self.assets_dir = assets_dir

        self.levels = load_boxoban_levels(level_pack_path)
        self.sprites = load_sprites(assets_dir, tile)

        self.show_goals = show_goals
        self.show_grid = show_grid
        self.show_hud = show_hud

        self.fig = plt.figure(figsize=(9.5, 10.8))
        gs = self.fig.add_gridspec(13, 1)
        self.ax = self.fig.add_subplot(gs[:10, 0])
        self.ctrl = self.fig.add_subplot(gs[10:, 0])
        self.ctrl.set_facecolor((0.95, 0.95, 0.95))
        self.ctrl.set_xticks([]); self.ctrl.set_yticks([])

        self._build_controls()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.timer = self.fig.canvas.new_timer(interval=int(1000/self.fps))
        self.timer.add_callback(self._on_timer)
        self.playing = False

        self._load_level(level_index)
        self._refresh()

    # ----- data -----

    def _build_states(self, level, first_state: State, moves: str):
        mlist = to_dirs(moves)
        states = [first_state]
        s = first_state
        for m in mlist:
            s = step(s, m)
            states.append(s)
        return mlist, states

    def _solve_level(self, idx: int) -> Optional[str]:
        try:
            return solve_sokoban(self.levels[idx], self.method)
        except Exception as e:
            print("Solve error:", e)
            return None

    def _load_level(self, idx: int):
        idx = max(0, min(idx, len(self.levels)-1))
        self.level_index = idx
        self.level, first_state = parse_level(self.levels[idx])
        self.moves = self._solve_level(idx) or ""
        self.move_list, self.states = self._build_states(self.level, first_state, self.moves)
        self.n = len(self.move_list)
        self.idx = 0
        self.pushes = sum(1 for m in self.move_list if m.isupper())

        frm = render_frame(self.level, self.states[self.idx], self.sprites, self.tile, self.show_goals)
        if hasattr(self, "im"):
            self.im.set_data(frm)
        else:
            self.im = self.ax.imshow(frm, extent=[0, self.level.width*self.tile, self.level.height*self.tile, 0])
        self.ax.set_axis_off()

        if hasattr(self, "grid_lines"):
            for ln in self.grid_lines: ln.remove()
        self.grid_lines = []
        for x in range(self.level.width+1):
            ln = self.ax.axvline(x*self.tile, lw=0.5, color="w", alpha=0.35, visible=self.show_grid)
            self.grid_lines.append(ln)
        for y in range(self.level.height+1):
            ln = self.ax.axhline(y*self.tile, lw=0.5, color="w", alpha=0.35, visible=self.show_grid)
            self.grid_lines.append(ln)

        if hasattr(self, "hud"):
            self.hud.remove()
        self.hud = self.ax.text(
            0.02, 0.02,
            "", color="w", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.45, boxstyle="round,pad=0.25"),
            transform=self.ax.transAxes, visible=self.show_hud
        )

        self._update_labels()
        self._refresh(redraw_only=True)

    # ----- controls -----

    def _build_controls(self):
        def A(x, y, w, h):
            l = self.ctrl.get_position()
            return self.fig.add_axes([l.x0 + x*l.width, l.y0 + y*l.height, w*l.width, h*l.height])

        self.ax_prev     = A(0.01, 0.35, 0.07, 0.55); self.btn_prev    = Button(self.ax_prev, "Prev")
        self.ax_play     = A(0.09, 0.35, 0.10, 0.55); self.btn_play    = Button(self.ax_play, "Play/Pause")
        self.ax_next     = A(0.20, 0.35, 0.07, 0.55); self.btn_next    = Button(self.ax_next, "Next")
        self.ax_restart  = A(0.28, 0.35, 0.08, 0.55); self.btn_restart = Button(self.ax_restart, "Restart")
        self.ax_save_gif = A(0.37, 0.35, 0.09, 0.55); self.btn_gif     = Button(self.ax_save_gif, "Save GIF")
        self.ax_save_mp4 = A(0.47, 0.35, 0.09, 0.55); self.btn_mp4     = Button(self.ax_save_mp4, "Save MP4")

        self.ax_fps = A(0.60, 0.42, 0.18, 0.42)
        self.slider = Slider(self.ax_fps, "", 1, 30, valinit=self.fps, valstep=1)  

        self.slider.label.set_text("FPS")
        self.slider.label.set_position((0.35, -0.55))         
        self.slider.label.set_horizontalalignment("center")
        self.slider.label.set_fontsize(10)
        self.slider.valtext.set_position((0.65, -0.55))       
        self.slider.valtext.set_horizontalalignment("center")
        self.slider.valtext.set_fontsize(10)
        self.slider.on_changed(self._on_speed)

        self.ax_chk = A(0.80, 0.12, 0.17, 0.76)
        self.check = CheckButtons(self.ax_chk, ["Grid", "Goals", "HUD"],
                                [self.show_grid, self.show_goals, self.show_hud])
        self.check.on_clicked(self._on_check)

        self.ax_file_btn = A(0.01, 0.05, 0.08, 0.22); self.btn_file = Button(self.ax_file_btn, "File…")

        self.ax_file_txt = A(0.10, 0.05, 0.31, 0.22)
        self.txt_file = TextBox(self.ax_file_txt, "", initial="", color=".95", hovercolor=".9")
        self.txt_file.text_disp.set_color("black")
        try: self.txt_file.cursor.set_visible(False)
        except Exception: pass

        self.ax_level_txt = A(0.43, 0.05, 0.10, 0.22)
        self.txt_level = TextBox(self.ax_level_txt, "Level", initial="0")

        self.ax_go = A(0.54, 0.05, 0.06, 0.22)
        self.btn_go = Button(self.ax_go, "Go")

        # callbacks
        self.btn_prev.on_clicked(lambda _e: self.prev_level())
        self.btn_next.on_clicked(lambda _e: self.next_level())
        self.btn_play.on_clicked(lambda _e: self.toggle())
        self.btn_restart.on_clicked(lambda _e: self.restart())
        self.btn_gif.on_clicked(lambda _e: self.save_gif())
        self.btn_mp4.on_clicked(lambda _e: self.save_mp4())
        self.btn_file.on_clicked(lambda _e: self.choose_file())
        self.btn_go.on_clicked(lambda _e: self.jump_level())

    # ----- handlers -----

    def _on_timer(self):
        if not self.playing:
            return
        if self.idx < self.n:
            self.idx += 1
            self._refresh()
        else:
            self.playing = False
            self.timer.stop()

    def _on_speed(self, val):
        self.fps = max(1, float(val))
        self.timer.interval = int(1000/self.fps)

    def _on_check(self, label):
        if label == "Grid":
            self.show_grid = not self.show_grid
            for ln in self.grid_lines:
                ln.set_visible(self.show_grid)
        elif label == "Goals":
            self.show_goals = not self.show_goals
        elif label == "HUD":
            self.show_hud = not self.show_hud
            self.hud.set_visible(self.show_hud)
        self._refresh(redraw_only=True)

    def _on_key(self, evt):
        if evt.key == " ":
            self.toggle()
        elif evt.key == "right":
            self.step_forward()
        elif evt.key == "left":
            self.step_back()
        elif evt.key == "r":
            self.restart()
        elif evt.key == "g":
            self.show_grid = not self.show_grid
            for ln in self.grid_lines: ln.set_visible(self.show_grid)
            self._refresh(redraw_only=True)
        elif evt.key == "h":
            self.show_hud = not self.show_hud
            self.hud.set_visible(self.show_hud)
            self._refresh(redraw_only=True)
        elif evt.key == "s":
            self.save_gif()
        elif evt.key in ("q", "escape"):
            plt.close(self.fig)

    # ----- actions -----

    def toggle(self):
        self.playing = not self.playing
        if self.playing: self.timer.start()
        else: self.timer.stop()

    def step_forward(self):
        self.playing = False; self.timer.stop()
        if self.idx < self.n:
            self.idx += 1
            self._refresh()

    def step_back(self):
        self.playing = False; self.timer.stop()
        if self.idx > 0:
            self.idx -= 1
            self._refresh()

    def restart(self):
        self.playing = False; self.timer.stop()
        self.idx = 0
        self._refresh()

    def next_level(self):
        self.playing = False; self.timer.stop()
        self._load_level(min(self.level_index + 1, len(self.levels)-1))

    def prev_level(self):
        self.playing = False; self.timer.stop()
        self._load_level(max(self.level_index - 1, 0))

    def jump_level(self):
        try:
            i = int(self.txt_level.text.strip())
        except Exception:
            print("Invalid level number.")
            return
        if not (0 <= i < len(self.levels)):
            print(f"Level out of range (0..{len(self.levels)-1}).")
            return
        self.playing = False; self.timer.stop()
        self._load_level(i)

    def choose_file(self):
        if _HAS_TK:
            try:
                root = _tk.Tk(); root.withdraw()
                p = _fd.askopenfilename(
                    title="Choose a Sokoban level text file",
                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
                )
                root.destroy()
            except Exception:
                p = ""
        else:
            p = ""
        if not p:
            print("No file chosen.")
            return
        try:
            self.levels = load_boxoban_levels(p)
            self.level_pack_path = p
            self._load_level(0)
        except Exception as e:
            print("Failed to load pack:", e)

    # ----- saving -----

    def _render_all_frames(self) -> List[Image.Image]:
        frames = []
        for i in range(self.n+1):
            frm = render_frame(self.level, self.states[i], self.sprites, self.tile, self.show_goals)
            frames.append(Image.fromarray(frm))
        return frames

    def save_gif(self):
        duration_ms = int(1000/self.fps)
        frames = self._render_all_frames()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"sokoban_replay_{ts}_lvl{self.level_index}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
        print(f"Saved GIF -> {out}")

    def save_mp4(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"sokoban_replay_{ts}_lvl{self.level_index}.mp4"
        frames = self._render_all_frames()
        try:
            import imageio.v3 as iio  
            arr = [np.asarray(f) for f in frames]
            iio.imwrite(out, arr, fps=max(1, int(self.fps)))
            print(f"Saved MP4 -> {out} (imageio-ffmpeg)")
            return
        except Exception as e:
            print("imageio-ffmpeg not available or failed:", e)
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=max(1, int(self.fps)))
            with writer.saving(self.fig, out, dpi=100):
                for i in range(self.n+1):
                    frm = render_frame(self.level, self.states[i], self.sprites, self.tile, self.show_goals)
                    self.im.set_data(frm)
                    writer.grab_frame()
            print(f"Saved MP4 -> {out} (matplotlib FFMpegWriter)")
        except Exception as e:
            print("FFMpeg export failed. Install ffmpeg or imageio-ffmpeg. Error:", e)

    # ----- draw -----

    def _update_labels(self):
        base = os.path.basename(self.level_pack_path)
        try:
            self.txt_file.set_val(f"{base}  (levels: {len(self.levels)})")
        except Exception:
            self.txt_file.text_disp.set_text(f"{base}  (levels: {len(self.levels)})")
            self.fig.canvas.draw_idle()
        self.txt_level.set_val(str(self.level_index))

    def _refresh(self, redraw_only: bool = False):
        frm = render_frame(self.level, self.states[self.idx], self.sprites, self.tile, self.show_goals)
        self.im.set_data(frm)
        if self.show_hud:
            mv = self.move_list[self.idx-1] if self.idx > 0 else "-"
            self.hud.set_text(
                f"Pack: {os.path.basename(self.level_pack_path)}   "
                f"Level: {self.level_index}/{len(self.levels)-1}   "
                f"Step: {self.idx}/{self.n}   Move: {mv}   Pushes: {self.pushes}"
            )
        self.fig.canvas.draw_idle()
        for ln in self.grid_lines:
            ln.set_zorder(self.im.get_zorder() + 1)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Replay solution on YOUR level using gym-sokoban textures with interactive controls."
    )
    ap.add_argument("--file", default="000.txt")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--method", default="astar", choices=["bfs", "dfs", "ucs", "astar"])
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--assets", default="")
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    level_path = os.path.normpath(os.path.join(here, "..", "boxoban_levels", args.file))

    assets_dir = args.assets or find_gym_sokoban_assets_dir()
    if not assets_dir:
        raise RuntimeError("Could not locate assets.")

    viewer = Viewer(level_pack_path=level_path, level_index=args.idx, method=args.method,
                    tile=args.tile, fps=args.fps, assets_dir=assets_dir)
    plt.show()

if __name__ == "__main__":
    main()

