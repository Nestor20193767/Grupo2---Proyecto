# Reporte de validación de la plantilla

Fecha: 2026-06-26

## Comprobaciones realizadas

- Compilación de `app/`, `src/`, `scripts/`, `tests/` y `legacy/`: **correcta**.
- Pruebas unitarias: **3 aprobadas**.
- Estructura mínima del repositorio: **correcta** con la excepción esperada de los binarios de modelo.
- Notebooks: formato JSON/nbformat 4 válido.
- Enlaces Markdown relativos: sin rutas rotas detectadas.
- Paper y figuras: incluidos.

## Acción pendiente obligatoria

Los binarios ya existentes en el GitHub original no se duplicaron dentro de esta plantilla. Antes de retirar `App/Modelo/`, ejecute `scripts/migrate_repository.py` para copiarlos a `models/` y luego:

```bash
python scripts/check_models.py --strict
python scripts/check_repository.py
```

La validación definitiva solo debe considerarse completa cuando ambos comandos terminen sin archivos faltantes.
