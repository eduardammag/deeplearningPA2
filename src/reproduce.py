"""Reproduce all required artifacts, then preserve a compact submission snapshot."""
from pathlib import Path
import hashlib
import json
import shutil
import platform
import torch
from .evaluation import save_metrics

def publish():
    required=["synthetic/results.json","synthetic/occlusion.png","synthetic/degradation.png",
        "comparison.json","decoupling.png","detector_comparison.json","ablation.json","ablation.png",
        "memory/results.json","memory/gradient.png","memory/empirical.png",
        "failures/case_1.png","failures/case_2.png","failures/case_3.png","failures/diagnoses.json",
        "correction.json","correction.png","stress.json","stress.png","inference.json"]
    for name in required:
        source=Path("outputs")/name
        if not source.exists(): raise FileNotFoundError("Missing required artifact: "+str(source))
    for source in Path("outputs").rglob("*"):
        if source.suffix not in (".png",".json"): continue
        if "ablation" in source.relative_to("outputs").parts[:-1]: continue
        if source.name.startswith("torchvision_"): continue
        destination=Path("results")/source.relative_to("outputs")
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
    files=list(Path("src").glob("*.py"))+[Path("metrics.py"),Path("train.py"),Path("pyproject.toml"),Path("checkpoints/gru.pt")]
    files+=list(Path("data_MOT17Labels/train").glob("*-FRCNN/gt/gt.txt"))
    files+=list(Path("data_MOT17Labels/train").glob("*-FRCNN/det/det.txt"))
    files+=list(Path("data_MOT17Labels/train/MOT17-09-FRCNN/img1").glob("*.jpg"))
    files+=list(Path(".cache/torch/checkpoints").glob("fasterrcnn_mobilenet_v3_large_320_fpn-*.pth"))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save_metrics(dict(python=platform.python_version(),torch=str(torch.__version__),device="cpu",
        hashes=hashes,required_artifacts=required),Path("results/manifest.json"))
    comparison=json.loads(Path("outputs/comparison.json").read_text())
    correction=json.loads(Path("outputs/correction.json").read_text())
    ablation=json.loads(Path("outputs/ablation.json").read_text())
    lines=["# Resultados para a apresentação","",
        "Valores executados; configuração principal GRU/T=16/seed=0, sem escolha pelo teste.","",
        "| Sequência | Baseline IDF1 | Temporal IDF1 | mAP |",
        "|---|---:|---:|---:|"]
    for row in comparison:
        lines.append(f"| {row['sequence']} | {row['baseline']['IDF1']:.4f} | {row['temporal']['IDF1']:.4f} | {row['mAP']:.4f} |")
    lines+=["","## Ablação na validação","","| Célula/janela | IDF1 média ± desvio |","|---|---:|"]
    for name,values in ablation["summary"].items():
        score=values["IDF1"]
        lines.append(f"| {name} | {score['mean']:.4f} ± {score['std']:.4f} |")
    memory=json.loads(Path("outputs/memory/results.json").read_text())
    lines += ["", "## Horizonte de memória", "",
              "| Célula | Redução da norma em 8 passos | Redução em 16 passos |",
              "|---|---:|---:|"]
    for cell,curve in memory["gradient"].items():
        if cell == "final": continue
        lines.append(f"| {cell} | {curve[0]/max(curve[8],1e-20):.1f}× | {curve[0]/max(curve[16],1e-20):.1f}× |")
    horizons=memory["empirical_horizons"]
    lines += ["",f"Nas {len(horizons)} trajetórias isoladas, o maior intervalo recuperado variou "
              f"de {min(horizons)} a {max(horizons)} quadros. A política mata tracks depois de três "
              "ausências; esse limite domina o horizonte empírico desta configuração.",
              "", "A ablação não mostrou uma quebra clara da RNN em IDF1 ao aumentar T. "
              "O gradiente se atenua fortemente em todas as células neste treino. Portas não garantem "
              "memória longa automaticamente. Na LSTM, a curva mede h, não o caminho independente de c."]
    stress=json.loads(Path("outputs/stress.json").read_text())
    lines += ["", "## Estresse sem retreino", "",
              "| Intensidade | mAP | Baseline IDF1 | Temporal IDF1 |",
              "|---|---:|---:|---:|"]
    for row in stress:
        lines.append(f"| {row['level']} | {row['mAP']:.4f} | {row['baseline']['IDF1']:.4f} | {row['temporal']['IDF1']:.4f} |")
    lines += ["", "As diferenças entre temporal e baseline são pequenas neste experimento: "
              "não há evidência de absorção relevante da degradação do detector. "
              "A perda de detecções e o ruído fragmentam ambas as soluções."]
    detectors=json.loads(Path("outputs/detector_comparison.json").read_text())
    lines += ["", "## Duas fontes de detecção, sequência 09 inteira", "",
              "| Fonte | mAP | IDF1 | ID switches |", "|---|---:|---:|---:|"]
    for name,row in detectors["detectors"].items():
        lines.append(f"| {name} | {row['mAP']:.4f} | {row['IDF1']:.4f} | {row['IDSW']} |")
    lines+=["","## Interpretação","",
        "A previsão geométrica recorrente não garante ganho: examine os valores acima. "
        "O treinamento por teacher forcing e a perda de caixa não otimizam diretamente identidade. "
        "Sob oclusão, o modelo realimenta previsões; erros acumulam e o portão de IoU pode rejeitar a observação correta.",
        "",correction["conclusion"],"",
        "A sobrevivência empírica inclui o limite max_missed=3. Não confundir esse limite de engenharia "
        "com desaparecimento do gradiente. A figura analítica mede estados ocultos; a empírica mede manutenção "
        "da identidade após oclusões injetadas em trajetórias reais isoladas.",
        "","As curvas e tabelas permitem verificar se há evidência de colapso da RNN simples. "
        "Se as médias forem próximas e seus desvios se sobrepuserem, não afirmar que a ablação demonstrou "
        "superioridade das portas; o horizonte de gradiente e a dificuldade dos dados precisam ser considerados.",
        "","Veja failures/diagnoses.json para os três erros concretos, visibilidade, IoU da previsão e figuras. "
        "As limitações do protocolo de métricas e a costura de janelas estão no README."]
    Path("results/APRESENTACAO.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def main():
    from .download import download_images
    from .analysis import main as experiments
    from .detector_experiment import main as detectors
    from .inference import infer_video
    download_images()
    experiments()
    detectors()
    result=infer_video("data_MOT17Labels/train/MOT17-09-FRCNN")
    save_metrics(result,"outputs/inference.json")
    publish()

if __name__=="__main__": main()
