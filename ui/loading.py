# ui/loading.py
import tkinter as tk
from tkinter import ttk

class LoadingOverlay:
    def __init__(self, parent, text="読み込み中..."):
        self.parent = parent
        self.text = text
        self.win = None
        self.pb = None

    def show(self):
        if self.win is not None:
            return
        self.win = tk.Toplevel(self.parent)
        self.win.title("")
        self.win.transient(self.parent)
        self.win.grab_set()  # モーダル化
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", lambda: None)  # 閉じられないように

        # 親の中央に配置
        self.win.update_idletasks()
        px = self.parent.winfo_rootx()
        py = self.parent.winfo_rooty()
        pw = self.parent.winfo_width()
        ph = self.parent.winfo_height()
        w, h = 280, 120
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.win.geometry(f"{w}x{h}+{x}+{y}")

        frame = ttk.Frame(self.win, padding=16)
        frame.pack(expand=True, fill="both")

        ttk.Label(frame, text=self.text, anchor="center").pack(pady=(4, 12))
        self.pb = ttk.Progressbar(frame, mode="indeterminate", length=220)
        self.pb.pack()
        self.pb.start(12)  # 速さ

        self.win.update_idletasks()

    def update_text(self, text: str):
        self.text = text
        if self.win:
            # ラベルを再取得して更新（1個目のラベル）
            for child in self.win.winfo_children():
                for g in child.winfo_children():
                    if isinstance(g, ttk.Label):
                        g.config(text=text)
                        break

    def close(self):
        if self.pb:
            try:
                self.pb.stop()
            except Exception:
                pass
        if self.win:
            try:
                self.win.grab_release()
            except Exception:
                pass
            self.win.destroy()
        self.win = None
        self.pb = None