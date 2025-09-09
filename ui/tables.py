# ui/tables.py
import tkinter as tk
from tkinter import ttk
import pandas as pd

class DualTablesView:
    @staticmethod
    def create_tree(parent, df: pd.DataFrame) -> ttk.Treeview:
        columns = list(df.columns) if not df.empty else []
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=10)

        yscroll = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        for col in columns:
            tree.heading(col, text=col)
            est_width = 140
            if not df.empty:
                try:
                    est_width = max(120, min(380, int(df[col].astype(str).str.len().mean() * 12)))
                except Exception:
                    pass
            tree.column(col, anchor="w", width=est_width, stretch=True)

        for _, row in df.iterrows():
            tree.insert("", "end", values=[row.get(c, "") for c in columns])

        tree.bind("<Configure>", lambda e: yscroll.place(in_=tree, relx=1.0, rely=0, relheight=1.0, anchor="ne"))
        xscroll.pack(fill="x", pady=(0,10))
        return tree

    @staticmethod
    def build_dual_tables(parent, top_df, bottom_df):
        container = ttk.Frame(parent)
        container.pack(expand=True, fill="both", padx=10, pady=10)

        ttk.Label(container, text="抽出5レコード（上段）").pack(anchor="w")
        top_tree = DualTablesView.create_tree(container, top_df)
        top_tree.pack(expand=True, fill="both", pady=(0,10))

        ttk.Label(container, text="他ファイル抽出コードと一致するレコード（下段）").pack(anchor="w")
        bottom_tree = DualTablesView.create_tree(container, bottom_df)
        bottom_tree.pack(expand=True, fill="both")