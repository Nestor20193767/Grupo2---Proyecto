# Modelos entrenados

Esta carpeta debe conservar los artefactos finales que actualmente se encuentran en [`App/Modelo/`](https://github.com/Nestor20193767/Grupo2---Proyecto/tree/main/App/Modelo) del repositorio original. Los archivos ONNX exportados con datos externos **no funcionan si se separan de su `.onnx.data` correspondiente**.

## Artefactos esperados

| Archivo | Uso | Requerido para la app |
|---|---|---:|
| `tcncvae_decoder_physionet.onnx` | Decoder generativo TCN-cVAE | Sí |
| `tcncvae_decoder_physionet.onnx.data` | Pesos externos del decoder | Sí |
| `clf_aug_physionet.onnx` | Clasificador TCN aumentado | Sí, para clasificación |
| `clf_aug_physionet.onnx.data` | Pesos externos del clasificador | Sí, para clasificación |
| `tcncvae_encoder_physionet.onnx` | Encoder del cVAE | No, solo análisis |
| `clf_aug_physionet.pt` | Checkpoint PyTorch | No, reentrenamiento |
| `label_encoder_physionet.pkl` | Orden de etiquetas del clasificador | Recomendado |
| `latent_bank.npz` | Distribuciones latentes por clase | Recomendado |

## Migración desde la estructura actual

```bash
python scripts/migrate_repository.py --source . --destination .
python scripts/check_models.py --strict
```

Para archivos grandes se recomienda Git LFS:

```bash
git lfs install
git lfs track "*.onnx" "*.onnx.data" "*.pt" "*.npz" "*.pkl"
git add .gitattributes models/
git commit -m "chore: versionar artefactos finales con Git LFS"
```

Como alternativa, publique una versión etiquetada en GitHub Releases o un registro de modelos y escriba el enlace y el SHA-256 en esta página. No deje enlaces temporales de Google Colab.
