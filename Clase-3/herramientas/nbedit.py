"""Helpers para editar notebooks ipynb con anclas estrictas."""
import json
import re


class NB:
    def __init__(self, path, label=""):
        self.path = path
        self.label = label or path
        self.nb = json.load(open(path, encoding="utf-8"))
        self.cells = self.nb["cells"]
        self.log = []

    # ---------- lectura ----------
    def src(self, i):
        return "".join(self.cells[i]["source"])

    def find(self, needle):
        """Devuelve los índices de celdas que contienen el texto."""
        return [i for i, c in enumerate(self.cells) if needle in self.src(i)]

    # ---------- escritura ----------
    @staticmethod
    def _to_source(text):
        lines = text.splitlines(keepends=True)
        if lines and lines[-1].endswith("\n"):
            lines[-1] = lines[-1][:-1]
        return lines

    def set(self, i, text, tag=""):
        self.cells[i]["source"] = self._to_source(text)
        self.log.append((i, tag or "set"))
        return i

    def rep(self, i, old, new, tag=""):
        s = self.src(i)
        n = s.count(old)
        assert n == 1, f"[{self.label}] celda {i}: ancla encontrada {n} veces (tag={tag}): {old[:70]!r}"
        self.set(i, s.replace(old, new), tag)
        return i

    def resub(self, i, pattern, new, tag=""):
        s = self.src(i)
        out, n = re.subn(pattern, lambda m: new, s, count=0, flags=re.S)
        assert n == 1, f"[{self.label}] celda {i}: regex matcheó {n} veces (tag={tag})"
        self.set(i, out, tag)
        return i

    def new_cell(self, cell_type, text, at=None):
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": self._to_source(text),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        if at is None:
            self.cells.append(cell)
        else:
            self.cells.insert(at, cell)
        self.log.append((at, f"new {cell_type}"))
        return cell

    def save(self, out):
        json.dump(self.nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return out
