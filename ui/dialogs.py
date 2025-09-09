# ui/dialogs.py
import tkinter as tk
from tkinter import ttk, simpledialog
from core.file_keys import PATIENT_CODE_CANDIDATES

class ColumnSelectDialog(simpledialog.Dialog):
    """ 患者コード列を選択させるダイアログ """
    def __init__(self, parent, columns, title="患者コード列を選択"):
        self.columns = columns
        self.selected = None
        super().__init__(parent, title)

    def body(self, master):
        ttk.Label(master, text="患者コードに相当する列を選択してください").grid(row=0, column=0, padx=10, pady=10)
        self.combo = ttk.Combobox(master, values=self.columns, state="readonly", width=40)
        self.combo.grid(row=1, column=0, padx=10, pady=(0,10))
        if self.columns:
            for cand in PATIENT_CODE_CANDIDATES:
                if cand in self.columns:
                    self.combo.set(cand)
                    break
            if not self.combo.get():
                self.combo.current(0)
        return self.combo

    def apply(self):
        self.selected = self.combo.get()