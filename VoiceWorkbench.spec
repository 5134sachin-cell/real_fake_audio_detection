# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata


def gather_tree(root: str, dest_root: str) -> list[tuple[str, str]]:
    root_path = Path(root)
    if not root_path.exists():
        return []

    items: list[tuple[str, str]] = []
    for path in root_path.rglob("*"):
        if path.is_file():
            relative_parent = path.relative_to(root_path).parent
            target_dir = Path(dest_root) / relative_parent
            items.append((str(path), str(target_dir)))
    return items


datas = []
datas += gather_tree("templates", "templates")
datas += gather_tree("static", "static")
datas += gather_tree("models", "models")
datas += gather_tree(".tts_cache", ".tts_cache")
datas += collect_data_files("TTS")
datas += copy_metadata("TTS")
datas += copy_metadata("torch")
datas += copy_metadata("numpy")
datas += copy_metadata("scipy")

binaries = []
binaries += collect_dynamic_libs("torch")

hiddenimports = [
    "TTS.api",
    "TTS.utils.manage",
    "TTS.utils.generic_utils",
    "TTS.utils.synthesizer",
    "TTS.tts.configs.vits_config",
    "TTS.tts.models.vits",
    "TTS.tts.layers.losses",
]


a = Analysis(
    ["product_launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoiceWorkbench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VoiceWorkbench",
)
