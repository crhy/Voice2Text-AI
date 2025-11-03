# PyInstaller spec file for Voice 2 Text
# Run: pyinstaller voice_app.spec

block_cipher = None

a = Analysis(
    ['voice_app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'faster_whisper',
        'torch',
        'torch.nn',
        'torch.optim',
        'scipy',
        'pygame',
        'PIL',
        'pyperclip',
        'pyaudio',
        'numpy',
        'requests',
        'gtts',
        'datetime',
        'queue',
        'json',
        'os',
        'tempfile',
        'wave',
        'threading',
        'time',
        'tkinter',
        'tkinter.ttk'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Voice2Text',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)