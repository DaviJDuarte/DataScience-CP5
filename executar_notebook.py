"""Executa o notebook com o Python deste processo e preserva todas as saídas."""
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from modelo import ROOT
from dados import salvar_json

if __name__ == "__main__":
    caminho = ROOT / "Checkpoint05_resolvido.ipynb"
    notebook = nbformat.read(caminho, as_version=4)
    reutilizou = (ROOT / "resultados/teste_final.json").exists()
    kernel = KernelManager(kernel_name="python3")
    kernel.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    cliente = NotebookClient(notebook, km=kernel, timeout=1800,
                             resources={"metadata": {"path": str(ROOT)}})
    try:
        cliente.execute()
    finally:
        nbformat.write(notebook, caminho)
    erros = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not erros
    salvar_json(ROOT / "resultados/execucao_notebook.json", {
        "celulas_codigo_executadas": sum(c.cell_type == "code" for c in notebook.cells),
        "erros": len(erros), "resultados_persistidos_reutilizados": reutilizou})
    print("Notebook executado integralmente, sem erros:", caminho)
