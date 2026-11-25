# Descarregar o projeto

Aceder ao repositório GitHub e clicar no botão verde Code → Download ZIP.

Extrair o ficheiro ZIP para uma pasta no computador.


# Reconstrução de RM com SigPy – demonstrador

Simula uma aquisição de RM acelerada: imagem → k-space → subamostragem → reconstrução
(zero-filled vs compressed sensing L1-wavelet) → métricas (PSNR/SSIM).

## Instalação
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Testado com Python 3.12 e sigpy 0.1.27. Não precisa de GPU (~1 min em CPU).

## Como correr
```bash
python run_demo.py          # gera figures/*.png e results/metrics.csv
jupyter notebook notebook.ipynb   # versão interativa passo a passo
```

## Conteúdo
| Ficheiro | Função |
|---|---|
| `run_demo.py` | Experiências 1–4 (k-space, aceleração, tipo de máscara, λ) |
| `mrisim.py` | Carregar fatia, máscaras, ruído, métricas |
| `notebook.ipynb` | Tutorial curto |
| `data/mni152_t1.nii.gz` | Template MNI ICBM152 T1 (2009a), fatia axial usada como ground truth |

## Parâmetros editáveis (topo de `run_demo.py`)
`SEED`, `NOISE` (ruído no k-space), `LAMBDA` (regularização). Fatia: `load_slice(z=...)`.

## Dataset e licença
O template MNI152 (ICBM 2009a Nonlinear Symmetric) vem incluído no pacote nilearn. Citar:
Fonov et al., NeuroImage 54(1), 2011; Fonov et al., NeuroImage 47, S102, 2009.
Confirma os termos de utilização antes de redistribuir publicamente.

## Limitações
- Simulação single-coil a partir de uma imagem magnitude (*inverse crime*): resultados mais otimistas que com dados reais.
- Subamostragem cartesiana 1D; sem mapas de sensibilidade (sem SENSE/ESPIRiT).
- Para dados reais multi-coil: fastMRI (requer registo) + `sigpy.mri.app.EspiritCalib` / `SenseRecon`.
